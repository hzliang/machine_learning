package com.anjuke.hzl.online.impl;

import com.anjuke.hzl.common.MathUtils;
import com.anjuke.hzl.common.OnlineOptimizeAlgorithmConst;
import com.anjuke.hzl.common.Pair;
import com.anjuke.hzl.online.OnlineOptimizeAlgorithm;
import org.apache.commons.csv.CSVRecord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.text.NumberFormat;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

/**
 * Created by huzuoliang on 2017/7/15.
 */
class FTRLP implements OnlineOptimizeAlgorithm {
    private final static Logger logger = LoggerFactory.getLogger(OnlineOptimizeAlgorithm.class);
    private final static NumberFormat nf = NumberFormat.getInstance();

    private String targetColumn;
    private String[] descriptiveColumns;
    private String[] numberColumns;
    private String[] categoricalColumns;

    private final Map<Integer, Double> z;
    private final Map<Integer, Double> n;

    private double targetRatio = 0;

    static {
        nf.setMaximumFractionDigits(2);
    }

    public FTRLP(Map<String, Object> meta) {

        this.targetColumn = meta.get("target").toString();
        this.numberColumns = (String[]) meta.get("numberColumns");
        this.descriptiveColumns = (String[]) meta.get("id");
        this.categoricalColumns = (String[]) meta.get("categorical");

        z = new HashMap<>();
        n = new HashMap<>();

    }

    @Override
    public double gradient4Loss(double predict, double really) {
        return 0;
    }

    @Override
    public double loss(double predictValue, double reallyValue) {

        final double p = Math.max(Math.min(predictValue, 1. - 10e-15), 10e-15);
        if (reallyValue == 1) {
            return -Math.log(p);
        }
        return -Math.log(1. - p);
    }

    public void update(double y, double p, Map<Integer, Double> x, Map<Integer, Double> w) {

        x.keySet().stream().forEach(item -> {
            double g = (p - y) * x.get(item);
            double s = (Math.sqrt(n.getOrDefault(item, 0.0) + g * g)
                    - Math.sqrt(n.getOrDefault(item, 0.0)))
                    / OnlineOptimizeAlgorithmConst.FTRL_ALPHA;
            z.put(item, z.getOrDefault(item, 0.0) + g - s * w.get(item));
            n.put(item, n.getOrDefault(item, 0.0) + g * g);
        });
    }

    private double logLikelihood = 0;

    @Override
    public void train(long index, Map<Integer, Double> x, int y) {
        final Map<Integer, Double> w = new HashMap<>();
        this.targetRatio = (1.0 * (index * this.targetRatio + y)) / (index + 1);
        double wtx = wtx(x, w);
        double p = 1. / (1. + Math.exp(-Math.max(Math.min(wtx, 35.), -35.)));
        logLikelihood += loss(p, y);

        if ((index + 1) % 100000 == 0) {
            logger.info("index = " + (index + 1) + ", loss = " + nf.format(logLikelihood));
        }
        this.update(y, p, x, w);
    }

    /**
     * w.transport * x
     *
     * @param x every line data
     * @return
     */
    private double wtx(Map<Integer, Double> x, Map<Integer, Double> w) {

        return x.keySet().stream().mapToDouble(hashIndex -> {
            if (Math.abs(z.getOrDefault(hashIndex, 0.0)) <= OnlineOptimizeAlgorithmConst.FTRL_L1) {
                w.put(hashIndex, 0.0);
            } else {
                int sign = MathUtils.sign(z.getOrDefault(hashIndex, 0.0));
                double wi = -(z.getOrDefault(hashIndex, 0.0) - sign * OnlineOptimizeAlgorithmConst.FTRL_L1)
                        / (OnlineOptimizeAlgorithmConst.FTRL_L2 + (OnlineOptimizeAlgorithmConst.FTRL_BETA
                        + Math.sqrt(n.getOrDefault(hashIndex, 0.0))) / OnlineOptimizeAlgorithmConst.FTRL_ALPHA);
                w.put(hashIndex, wi);
            }

            return w.get(hashIndex) * x.get(hashIndex);
        }).sum();
    }

    @Override
    public List<Pair> predictProbability(Iterable<CSVRecord> records) {

        return StreamSupport.stream(records.spliterator(), true).map(item -> {
            final String id = item.get("id");
            final Map<Integer, Double> x = getLine(item);
            final double wtx = wtx(x, new HashMap<>());
            return new Pair(id,1. / (1. + Math.exp(-Math.max(Math.min(wtx, 35.), -35.))));
        }).collect(Collectors.toList());
    }

    @Override
    public int predictClass(Map<Integer, Double> x) {
//        double p = predictProbability(x);
//        return p < this.targetRatio ? -1 : 1;
        return 0;
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

    private Map<Integer, Double> getLine(CSVRecord record) {
        Map<Integer, Double> x = new HashMap<>();

        final int numCols = numberColumns.length;
        for (int i = 0; i < numCols; i++) {
            double value = Double.parseDouble(record.get(numberColumns[i]));
            x.put(i, value);
        }
//            for (int i = 0; i < categoricalColumns.length; i++) {
//                String value = record.get(categoricalColumns[i]);
////                int idx = doHash(item + "_" + value, OnlineOptimizeAlgorithmConst.MAX_FEATURES) + numCols;
//                int idx = i;
//                if (x.containsKey(idx)) {
//                    x.put(idx, x.get(idx) + 1);
//                } else {
//                    x.put(idx, 1.0);
//                }
//            }
        Stream.of(categoricalColumns).forEach(item -> {
            String value = record.get(item);
            int hashIndex = hash(item + "_" + value, OnlineOptimizeAlgorithmConst.MAX_FEATURES) + numCols;
            x.put(hashIndex, x.getOrDefault(hashIndex, 0.0) + 1);
        });
        return x;
    }

    public int hash(String item, int maxFeatures) {
        return Math.abs(item.hashCode()) % maxFeatures;
    }
}
