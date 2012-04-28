(ns random-forests.test.train
  (:use [clojure.test])
  (:require [random-forests.train :as rf]))


(deftest test-parse-target
  (is "foo" (rf/parse-target "foo=continuous")))

(deftest test-target-index
  (is 1 (rf/target-index ["abcd" "foo"] "foo")))

(deftest test-parse-examples
  (let [header ["abcd" "foo"]
        input1  [["a" "2.0"]]
        input2  [["a" "NA"]]
        target-index 1
        encoders (rf/encoding-fns (list "abcd=categorical" "foo=continuous"))
        output1 [["a" 2.0]]
        output2 [["a" nil]]]
    (is output1 (rf/parse-examples header input1 target-index encoders))
    (is output2 (rf/parse-examples header input2 target-index encoders))))



