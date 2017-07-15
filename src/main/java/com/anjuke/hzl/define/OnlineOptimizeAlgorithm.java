package com.anjuke.hzl.define;

/**
 * Created by huzuoliang on 2017/7/15.
 */
public interface OnlineOptimizeAlgorithm {
    void train();
    void predictProbability();
    void predictClass();
}
