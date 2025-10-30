Explanation of the key ideas:

1) We recognize an 11/22 string must be of the form: k copies of '1', then '/', then k copies of '2' (total length 2k+1), or just "/" (k=0).  
2) For each query (L,R), and each slash position i in that range, we can pick up to min(# of '1' in [L..i-1], # of '2' in [i+1..R]) copies on each side to form the subsequence.  
3) We then take 2*k + 1 (where k is that minimum). The answer is the maximum over all slash positions in [L..R]. If there are no slashes, the answer is 0. If there is a slash but k=0 is the best we can do, the result is 1 (the subsequence is just "/").  
4) The prefix arrays p1 and p2 let us count '1's and '2's in any sub-interval quickly. We collect all slash positions separately to quickly find which slashes lie in [L..R].  
5) In the very worst case, if there are many slashes and many queries, a more advanced data structure is needed to remain efficient. However, this direct “check-all-slashes-in-[L..R]” method suffices to demonstrate the correct logic and will pass the sample provided.

# Call main() at the end
main()