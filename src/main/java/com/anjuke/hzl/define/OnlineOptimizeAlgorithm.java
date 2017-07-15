package com.anjuke.hzl.define;

import java.util.Map;

/**
 * Created by huzuoliang on 2017/7/15.
 */
public interface OnlineOptimizeAlgorithm {
    double loss(double predict,double really);
    void train(long index, Map<Long, Double> data, int target);
    double predictProbability(Map<Long, Double> x);
    int predictClass(Map<Long, Double> x);
}
