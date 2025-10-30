Explanation of the core formula used:

1) Let invAll = total inversions of P.  
2) For each subarray of length K, let invS_i = the number of inversions entirely inside that subarray in the original permutation.  
3) When that subarray is uniformly permuted, its internal inversions contribute on average K*(K-1)/4.  
4) It can be shown (though it is somewhat involved) that the expected change in cross-subarray inversions is zero when averaging over all i.  
   Intuitively, randomizing a block of K distinct numbers does not, on average, create or destroy cross-inversions with the rest of the array.  
5) Hence the expected inversion count (after picking i uniformly and shuffling that subarray) is:  
     (1/(N-K+1)) × Σ [ invAll - invS_i + K*(K-1)/4 ]  
   = invAll + (K*(K-1)/4) - (1/(N-K+1)) × Σ invS_i.  
6) We compute each piece in O(N log N) and combine them in modular arithmetic.

# End.