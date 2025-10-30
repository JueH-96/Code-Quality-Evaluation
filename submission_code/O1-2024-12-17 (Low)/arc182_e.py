Explanation of the key steps (high level):
1) We note that the expression (Ck + A_i) mod M has period P = M / gcd(M, C) in k.  
2) When gcd(C, M) = 1, that set of shifts (Ck mod M) for k=0..M-1 is simply all residues 0..M-1.  
   When gcd(C, M) = g > 1, the set of shifts is every g-th residue 0, g, 2g, …, (P-1)*g.  
3) We first construct a piecewise representation of h(r) = min_i((r + A_i) mod M) for r=0..M-1.  
   This relies on sorting the A_i and performing a "sweep" over breakpoints where each line  
   (r + x_i) would wrap from (r + x_i) to (r + x_i - M).  
4) Having that piecewise function, we can quickly sum h(r) over all r that are multiples of g  
   (the sub-lattice). That sum is the "blockSum" for one period in k.  
5) Finally we multiply blockSum by (K // P) and add the partial sum for the leftover (K % P) terms,  
   which again are the first leftover multiples of g in that sub-lattice.  
6) Printing the resulting sum solves the problem.

# do not forget to call main()
main()