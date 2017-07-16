package com.anjuke.hzl.common;

/**
 * Created by huzuoliang on 2017/7/16.
 */
public class Pair {
    private String key;
    private Object value;

    public Pair(String key, Object value){
        this.key = key;
        this.value = value;
    }

    @Override
    public String toString(){
        return key+","+value;
    }
}
