package com.anjuke.hzl.common;

/**
 * Created by huzuoliang on 2017/7/16.
 */
public class MathUtils {

    public static int sign(double d){
        if(d > 0){
            return 1;
        }else if(d < 0){
            return -1;
        }
        return 0;
    }
}
