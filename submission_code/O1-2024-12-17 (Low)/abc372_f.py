Explanation of the main ideas (high level):

1) We label as “important” (trackSet) all vertices that either appear as an endpoint of an extra edge (X_i or Y_i) or are vertex 1 (the start), plus their immediate successor on the cycle.  
2) We maintain a small DP array dpIn(k) of size len(trackSet) that tells how many ways to be in each tracked vertex at step k.  
3) The total number of walks of length k, s(k), evolves by the recurrence s(k+1) = s(k) + ∑(degM(v)*dpIn(k)[v]) over v in trackSet, because each tracked vertex v has degM(v) “extra” out-edges. Vertices not in trackSet have no extra out-edges, so they contribute nothing to that sum.  
4) To update dpIn for the next step, each vertex u in trackSet collects contributions from all predecessors w in trackSet that have either w -> u as a cycle edge (if w+1 = u) or w -> u as an extra M-edge. Outside vertices never feed back in, so we only need to consider edges internal to trackSet.  
5) We iterate this K times. The array dpIn is at most ~200 in length, and M ≤ 50, so each iteration can be done in O(|trackSet| + M) ≈ O(250). Thus the overall O(K*(|trackSet|+M)) ~ 2×10^5 × 250 = 5×10^7, which is high but can often be done carefully in optimized Python (and is typical for C++).  
6) Finally, s(K) is the total number of walks of length K. We output it modulo 998244353.  

# Do not forget to call main()!
main()