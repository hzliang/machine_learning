package com.anjuke.hzl.impl;

import com.anjuke.hzl.common.MathUtils;
import com.anjuke.hzl.common.OnlineOptimizeAlgorithmConst;
import com.anjuke.hzl.define.OnlineOptimizeAlgorithm;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by huzuoliang on 2017/7/15.
 */
public class FTRLP implements OnlineOptimizeAlgorithm {

    private Map<Long, Double> z = new HashMap<>();
    private Map<Long, Double> n = new HashMap<>();
    private Map<Long, Double> w = new HashMap<>();
    private double targetRatio = 0;

    @Override
    public double loss(double predictValue, double reallyValue) {

        final double p = Math.max(Math.min(predictValue, 1. - 10e-15), 10e-15);
        if (reallyValue == 1) {
            return -Math.log(p);
        }
        return -Math.log(1. - p);
    }

    public void update(double reallyValue, double predictValue, Map<Long, Double> x, Map<Long, Double> w) {

        x.keySet().stream().forEach(i -> {
            double g = (predictValue - reallyValue) * x.get(i);
            double s = (Math.sqrt(n.get(i) + g * g) - Math.sqrt(n.get(i)))
                    / OnlineOptimizeAlgorithmConst.FTRL_ALPHA;
            z.put(i, z.get(i) + g - s * w.get(i));
            n.put(i, n.get(i) + g * g);
        });
    }

    @Override
    public void train(long index, Map<Long, Double> x, int target) {

        this.targetRatio = (1.0 * (index * this.targetRatio + target)) / (index + 1);
        double wtx = wTX(x);
        double p = 1. / (1. + Math.exp(-Math.max(Math.min(wtx, 35.), -35.)));
        this.update(target, p, x, w);
    }

    private double wTX(Map<Long, Double> x) {

        w.clear();
        return x.keySet().stream().mapToDouble(indx -> {

            if (Math.abs(z.get(indx)) <= OnlineOptimizeAlgorithmConst.FTRL_L1) {
                w.put(indx, 0.0);
            } else {
                int sign = MathUtils.sign(z.get(indx));

                double temp = -(z.get(indx) - sign * OnlineOptimizeAlgorithmConst.FTRL_L1)
                        / (OnlineOptimizeAlgorithmConst.FTRL_L2 + (OnlineOptimizeAlgorithmConst.FTRL_BETA
                        + Math.sqrt(n.get(indx))) / OnlineOptimizeAlgorithmConst.FTRL_ALPHA);
                w.put(indx, w.get(indx) - temp);
            }

            return w.get(indx) * x.get(indx);
        }).sum();
    }

    @Override
    public double predictProbability(Map<Long, Double> x) {
        double wtx = wTX(x);
        double p = 1. / (1. + Math.exp(-Math.max(Math.min(wtx, 35.), -35.)));
        return p;
    }

    @Override
    public int predictClass(Map<Long, Double> x) {
        double p = predictProbability(x);
        return p < this.targetRatio ? -1 : 1;
    }
}
