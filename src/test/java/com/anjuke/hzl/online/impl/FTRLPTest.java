package com.anjuke.hzl.online.impl;

import com.anjuke.hzl.common.Pair;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;

import java.io.*;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by huzuoliang on 2017/7/16.
 */
public class FTRLPTest {
    public static void main(String[] args) throws IOException {

        String[] strs = new String[]{"id", "hour", "C1", "banner_pos"
                , "site_id", "site_domain", "site_category", "app_id", "app_domain", "app_category", "device_id"
                , "device_ip", "device_model", "device_type", "device_conn_type", "C14", "C15", "C16", "C17"
                , "C18", "C19", "C20", "C21"};

        Map<String, Object> meta = new HashMap<>(3);
        meta.put("targetColumn", "click");
        meta.put("idColumn", "id");
        meta.put("numberColumns", new String[]{});
        meta.put("categoricalColumns", strs);
        String path = "/root/calc.csv";
        FTRLP ftrlp = new FTRLP(meta,0.9,1,0.03,1);
        ftrlp.fit(path);

        path = "/root/test.csv";
        FileWriter fw = new FileWriter("opt.csv");
        fw.write(ftrlp.forKaggle(path));
        fw.close();
    }
}
