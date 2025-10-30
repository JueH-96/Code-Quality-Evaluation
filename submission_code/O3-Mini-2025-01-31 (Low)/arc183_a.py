def main():
    import sys, math
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    K = int(data[1])
    total = N * K  # length of the sequence

    # The number of good sequences is the multinomial coefficient:
    #   S = (N*K)! / (K!)^N
    # Python’s math.factorial works in C and is efficient enough even for numbers around 250000!
    # (The worst-case N*K = 500*500 = 250000)
    fact_total = math.factorial(total)
    fact_K = math.factorial(K)
    W = fact_total // (fact_K ** N)
    
    # We need to output the floor((S+1)/2)-th sequence in lexicographical order.
    T = (W + 1) // 2

    # We now use an iterative “greedy” procedure. At each step, we decide which number to choose.
    #
    # Suppose our current multiset (counts for each number 1..N) yields "current_ways" sequences.
    # When there are r positions left, if we choose a number x (with count > 0), the number
    # of completions is:
    #      branch = current_ways * (counts[x]) // r
    # This recurrence follows because if you view the multinomial formula
    #   current_ways = (r)! / (prod(count_i!))
    # then after taking x (decreasing its count by 1) the number of completions becomes
    #   (r-1)! / ( (counts[x]-1)! * (other factorials) )
    # and the ratio of the two is exactly counts[x] / r.
    #
    # Hence we can avoid recomputing factorials at every step.
    
    res = []              # The result sequence.
    r = total             # remaining positions.
    current_ways = W      # number of completions from the current state.
    counts = [K] * (N + 1)  # 1-indexed counts; counts[1..N] each equal to K initially.

    for _ in range(total):
        # Consider candidates in increasing order (to maintain lex order).
        for x in range(1, N+1):
            if counts[x] == 0:
                continue
            # If we decide to use x, then the number of completions of the sequence becomes:
            branch = current_ways * counts[x] // r
            # If T is greater than branch, then the T-th sequence is not among those starting with x.
            if T > branch:
                T -= branch
            else:
                # We choose x as the next element in our sequence.
                res.append(x)
                # Update the current number of completions to what remains after our choice.
                current_ways = branch
                counts[x] -= 1
                r -= 1
                break

    sys.stdout.write(" ".join(map(str, res)))
    
if __name__ == "__main__":
    main()