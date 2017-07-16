package com.anjuke.hzl.online;

import com.anjuke.hzl.common.Pair;
import org.apache.commons.csv.CSVRecord;

import java.util.List;
import java.util.Map;

/**
 * Created by huzuoliang on 2017/7/15.
 */
public interface OnlineOptimizeAlgorithm {
    double gradient4Loss(double predict,double really);
    double loss(double predict,double really);
    void train(long index, Map<Integer, Double> x, int target);
    List<Pair> predictProbability(Iterable<CSVRecord> records);
    int predictClass(Map<Integer, Double> x);
}
