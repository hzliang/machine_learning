package com.anjuke.hzl.online;

import com.anjuke.hzl.common.Pair;
import com.anjuke.hzl.common.TrickStatus;
import org.apache.commons.csv.CSVRecord;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

/**
 * Created by huzuoliang on 2017/7/15.
 */
public abstract class OnlineOptimizeAlgorithm {

    protected String targetColumn;
    protected String[] descriptiveColumns;
    protected String[] numberColumns;
    protected String[] categoricalColumns;

    protected double targetRatio = 0;
    protected double logLikelihood = 0;

    protected long status;

    protected OnlineOptimizeAlgorithm(Map<String, Object> meta) {

        this.targetColumn = meta.get("target").toString();
        this.numberColumns = (String[]) meta.get("numberColumns");
        this.descriptiveColumns = (String[]) meta.get("id");
        this.categoricalColumns = (String[]) meta.get("categorical");

        this.status = TrickStatus.toLong(1,1,1,100000,10000000,1);
    }

    public OnlineOptimizeAlgorithm(Map<String, Object> meta, long status) {

        this(meta);
        this.status = status;
    }

    protected abstract double probalityFunction(double wtx);

    protected abstract double lossGradientFunction(double p, double y, double xi);

    protected abstract double lossFunction(double p, double y);

    protected abstract void train(long index, Map<Integer, Double> x, int target);

    protected abstract void update(double y, double p, Map<Integer, Double> x, Map<Integer, Double> w);

    protected abstract double wtx(Map<Integer, Double> x, Map<Integer, Double> w);

    public List<Pair> predictProbability(Iterable<CSVRecord> records) {
        return StreamSupport.stream(records.spliterator(), true).map(item -> {
            final String id = item.get(targetColumn);
            final Map<Integer, Double> x = getLine(item);
            final double wtx = wtx(x, new HashMap<>());
            return new Pair(id, probalityFunction(wtx));
        }).collect(Collectors.toList());
    }

    protected abstract int predictClass(Map<Integer, Double> x);

    /**
     * get input
     * judge if use hash trick for save memeory
     * @param record
     * @return
     */
    private Map<Integer, Double> getLine(CSVRecord record) {

        Map<Integer, Double> x = new HashMap<>();
        for (int i = 0; i < numberColumns.length; i++) {
            double value = Double.parseDouble(record.get(numberColumns[i]));
            x.put(i, value);
        }
        if (TrickStatus.useHashTrick(status)) {
            Stream.of(categoricalColumns).forEach(item -> {
                String value = record.get(item);
                int hashIndex = hash(item, value) + numberColumns.length;
                x.put(hashIndex, x.getOrDefault(hashIndex, 0.0) + 1);
            });
            return x;
        }
        for (int i = 0; i < categoricalColumns.length; i++) {
            x.put(i, x.getOrDefault(i, 0.0) + 1);
        }
        return x;
    }

    /**
     * hash for key+"_"+value
     * @param key
     * @param value
     * @return
     */
    private int hash(String key, String value) {
        return Math.abs((key+"_"+value).hashCode()) % TrickStatus.getMaxFeatures(status);
    }

    public void fit(Iterable<CSVRecord> records) throws IOException {

        int t = 0;
        for (CSVRecord record : records) {
            final int y = Integer.parseInt(record.get(this.targetColumn));
            Map<Integer, Double> x = getLine(record);
            train(t, x, y);
            t++;
        }
    }
}
