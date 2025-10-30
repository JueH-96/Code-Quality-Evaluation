Explanation of the core idea:

1) Read N (the number of parts in the ring) and Q (the number of instructions).  
2) Keep track of the current positions of the left hand (Lpos) and the right hand (Rpos), initially (1, 2).  
3) For each instruction (hand H, target T):  
   • If the specified hand H is already at T, do nothing (0 cost).  
   • Otherwise, if T is occupied by the other hand, we must move that other hand exactly one step to an adjacent free part (cost +1).  
   • Then move the specified hand from its current position to T by the ring distance (the minimum of forward or backward distance on the ring).  
4) Sum all these move costs and output the total.

This procedure matches the examples given in the problem statement and is efficient (O(Q)) because N ≥ 3 guarantees there is always at least one spare part to which we can move the other hand by one step if necessary. At the end, we print the total cost of all such moves.

# Call main() at the end
if __name__ == "__main__":
    main()