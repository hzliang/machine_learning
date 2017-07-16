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
        meta.put("target", "click");
        meta.put("numberColumns", new String[]{});
        meta.put("id", new String[]{});
        meta.put("categorical", strs);
        String path = "/Users/huzuoliang/train.csv";
        Reader in = new FileReader(path);
        Iterable<CSVRecord> records = CSVFormat.RFC4180.withFirstRecordAsHeader().parse(in);
        FTRLP ftrlp = new FTRLP(meta);
        ftrlp.fit(records);

        path = "/Users/huzuoliang/test.csv";
        in = new FileReader(path);
        records = CSVFormat.RFC4180.withFirstRecordAsHeader().parse(in);
        List<Pair> ps = ftrlp.predictProbability(records);
        StringBuilder sb =new StringBuilder();
        for (Pair p : ps) {
            sb.append(p.toString()+"\n");
        }
        FileWriter fw = new FileWriter("opt.csv");
        fw.write(sb.toString());
        fw.close();
    }
}
