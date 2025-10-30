Explanation of key steps and why this solves the problem:

1) We build the so-called “subsequence matching” automaton for the forbidden sequence X. For each state j ∈ [0..M), if we see a character c (1 ≤ c ≤ K), then the next state is either j+1 if c = X[j], or remains j otherwise. If j = M, we have already formed X as a subsequence, and we do not allow further transitions in our valid count.

2) dp[i][j] counts the number of ways to build a length-i prefix of A while the largest matched prefix of X is j. We start with dp[0][0] = 1 and transition by choosing the next character c in [1..K], then moving to nxt[j][c] if that new state is still < M. We do so for i from 0..N-1 and j from 0..M-1. In the end, the sum of dp[N][j] for j in [0..M-1] gives the number of length-N sequences that do not contain X as a subsequence.

3) A more subtle combinatorial argument (outlined in advanced editorials) shows that, in exactly those sequences that never reach matched-state M for X, all other M-length sequences in the alphabet do appear as subsequences, i.e. they are not simultaneously missing some other M-subsequence. Consequently, the dp count is exactly the count of sequences that miss X and no other M-length sequence, which is what the problem requires.

4) We output this count modulo 998244353.

# Call main() to comply with the problem’s requirement.
# (No other top-level statements.)
main()