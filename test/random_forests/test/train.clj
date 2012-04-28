(ns random-forests.test.train
  (:use [clojure.test])
  (:require [random-forests.train :as rf]))

(deftest test-parse-target
  (is (= "foo" (rf/parse-target "foo=continuous"))))

(deftest test-target-index
  (is (= 1 (rf/target-index ["abcd" "foo"] "foo"))))

(deftest test-parse-examples
  (let [header ["abcd" "foo"]
        input1  [["a" "2.0"]]
        input2  [["a" "NA"]]
        input3  [["a" "2.0"]]
        target-index 1
        encoders (rf/encoding-fns (list "abcd=categorical" "foo=continuous"))
        output1 [["a" 2.0 2.0]]
        output2 (list)
        output3 [["a" 2.0]]]
    (is (= output1 (rf/parse-examples header input1 encoders target-index)))
    (is (= output2 (rf/parse-examples header input2 encoders target-index)))
    (is (= output3 (rf/parse-examples header input3 encoders)))))
