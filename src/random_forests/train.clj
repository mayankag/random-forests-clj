(ns random-forests.train
  (:use [clojure.tools.cli :only (cli)]
        [random-forests.encoding :only (text-tokens)])
  (:require [clojure-csv.core :as csv]
            [random-forests.core :as rf])
  (:gen-class :main true))

(defn named-examples
  "converts a header row and collection if rows to name-value pairs"
  [header input]
  (->> input
       (map #(map vector header %))))

(defn encoder-fn
  "returns encoder function for feature of type kind"
  [kind]
  (case kind
    "text"       (fn [x] (set (text-tokens x)))
    "continuous" (fn [x]
                   (try (Double/parseDouble x)
                        (catch java.lang.NumberFormatException e nil)))
    identity))

(defn csv-from-path
  "reads csv file from path"
  [path]
  (-> path
      slurp
      csv/parse-csv))

(defn encoding-fns
  "returns map of feature name to encoding-fn"
  [feature-desc]
  (->> feature-desc
       (map #(clojure.string/split % #"="))
       (map (fn [[name kind]] [name (encoder-fn kind)]))
       (into {})))

(defn features
  "returns collection of features from feature description"
  [header feature-desc]
  (let [header (->> header
                    (map vector (iterate inc 0))
                    (map reverse)
                    (map vec)
                    (into {}))]
    (->> feature-desc
       (map #(clojure.string/split % #"="))
       (map (fn [[name kind]] (rf/feature name (header name) (keyword kind)))))))

(defn auc-loss
  "measures auc loss from forest evaluation"
  [evaluation]
  (rf/auc evaluation))

(defn mean-absolute-loss
  "measures l1 loss from forest evaluation"
  [evaluation]
  (->> evaluation
       (map (fn [[a b]] (Math/abs (- a b))))
       (rf/avg)))

(defn mean-classification-error
  "measures l1 loss from forest evaluation"
  [evaluation]
  (->> evaluation
       (map (fn [[a b]] (if (= a b) 1 0)))
       (rf/avg)
       (float)))

(defn parse-target
  [input]
  (-> (clojure.string/split input #"=") first))

(defn target-index
  [header target-name]
  (->> header
       (keep-indexed (fn [i x] (if (= x target-name) i)))
       (first)))

(defn parse-examples
  ([header input encoding]
     (->> (named-examples header input)
          (map #(map (fn [[name val]]
                       ((get encoding name identity)  val)) %))
          (map vec)))
  ([header input encoding target-index]
     (->> (parse-examples header input encoding)
          (map (fn [z] (conj z (nth z target-index))))  ;; target is at end
          (filter #(not (nil? (last %)))))))

(defn write-output
  [options evaluation trees]
  (spit (:output options)
        (->> evaluation
             (map (fn [[a b]] [(count trees) a b (if (:multi options) (if (= a b) 1 0) (- a b))]))
             (map #(str (clojure.string/join "," %) "\n"))
             (reduce str))
        :append true))

(defn write-test-csv
  [options trees test-examples test-header target-name]
  (let [num-trees (count trees)
        output-file (str "prediction-" target-name "-" num-trees ".csv")]
    (spit output-file
          (str
           (clojure.string/join "," (conj test-header target-name)) "\n"
           (->> test-examples
               (map #(rf/votes trees %))
               (map rf/avg)
               (map #(conj %1 %2) test-examples)
               (map #(str (clojure.string/join "," %) "\n"))
               (reduce str))))))

(defn -main
  [& args]
  (let [[options args banner] (cli args
                                   ["-h" "--help" "Show help" :default false :flag true]
                                   ["-f" "--features" "Features specification (matching CSV header): name=continuous,foo=text" :parse-fn #(clojure.string/split % #",") :default []]
                                   ["-s" "--size" "Size of bootstrap sample per tree" :parse-fn #(Integer/parseInt %) :default 1000]
                                   ["-m" "--split" "Number of features to sample for each split" :parse-fn #(Integer/parseInt %) :default 100]
                                   ["-o" "--output" "Write detailed training error output in CSV format to output file"]
                                   ["-t" "--target" "Prediction target name"]
                                   ["-b" "--binary" "Perform binary classification of target (measures AUC loss)" :default false :flag true]
                                   ["-u" "--multi" "Perform multi-class classification of target (measures classification rate)" :default false :flag true]
                                   ["-l" "--limit" "Number of trees to build" :parse-fn #(Integer/parseInt %) :default 100]
                                   ["-q" "--test" "test file"])]
    (when (or (not (first args)) (:help options))
      (println banner)
      (System/exit 0))
    (let [input            (csv-from-path (first args))
          [test-header & test-input]     (when (:test options)
                                           (csv-from-path (:test options)))
          [header & input] input
          encoding         (encoding-fns (conj (:features options) (:target options)))
          target-name      (parse-target (:target options))
          target-index     (target-index header target-name)
          examples         (parse-examples header input encoding target-index)
          test-examples    (when test-input
                             (parse-examples header test-input encoding))
          features         (set (features header (:features options)))]
      (let [forest      (take (:limit options)
                              (rf/build-random-forest examples features (:split options) (:size options)))
            sub-forests (->> (range 1 (inc (:limit options)))
                             (map #(take % forest)))]
        (if (:output options)
          (spit (:output options) "tree_count,target,prediction,error\n"))
        (doseq [trees sub-forests]
          (let [combiner   (if (:multi options) rf/mode rf/avg)
                evaluation (rf/evaluate-forest trees combiner)
                loss       (-> evaluation
                               ((if (:binary options) auc-loss (if (:multi options) mean-classification-error mean-absolute-loss))))]
            (println (format "%d: %f" (count trees) loss))
            ;; write a test file if present
            (when test-input
              (write-test-csv options trees test-examples test-header target-name))
            (when (:output options)
              (write-output options evaluation trees)))))
      (shutdown-agents))))
