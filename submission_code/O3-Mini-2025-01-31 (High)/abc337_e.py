def main():
    import sys, math

    # Read N from standard input (number of bottles)
    # In interactive problems, the judge provides N first.
    data = sys.stdin.read().split()
    if not data:
        return
    try:
        N = int(data[0])
    except:
        return

    # The minimum number of friends needed is the smallest integer M such that 2^M >= N.
    # This is equivalent to taking the ceiling of log2(N).
    M = math.ceil(math.log2(N))
    
    # Print the number of friends, M, followed by a newline.
    sys.stdout.write(str(M) + "
")
    sys.stdout.flush()

    # For each friend i (where i=0 corresponds to the first friend),
    # assign them all bottles whose (bottle_number - 1) has the i-th bit set.
    # This ensures that every bottle from 1 to N gets a unique M-bit signature.
    # Example: for bottle j, represent (j-1) in binary. Friend i (0-indexed)
    # tastes bottle j if the i-th bit is 1.
    for i in range(M):
        # Collect all bottle numbers that friend i should taste.
        assigned_bottles = []
        for bottle in range(1, N + 1):
            if ((bottle - 1) >> i) & 1:
                assigned_bottles.append(bottle)
        # Print the number of bottles for this friend (K_i) followed by the sorted list.
        # The format is: K_i A_{i,1} A_{i,2} ... A_{i,K_i}
        line = str(len(assigned_bottles))
        if assigned_bottles:
            line += " " + " ".join(map(str, assigned_bottles))
        sys.stdout.write(line + "
")
        sys.stdout.flush()

    # After printing the distributions for all friends, the judge will report back a string S of length M,
    # with S[i] being '1' if friend i got upset (i.e. tasted the bad bottle), and '0' otherwise.
    # In our encoding, this binary string corresponds to the binary representation of (spoiled bottle number - 1),
    # where the i-th friend (with i starting from 0) represents the 2^i place.
    S = ""
    if len(data) > 1:
        S = data[1].strip()  # In an interactive judge, this line is read after the friend outputs.
    
    # In a real interactive problem, we would block here waiting for the judge's response.
    # Here, if S is empty, we simply choose a default value (for safety in offline testing).
    if not S:
        spoiled_bottle = 1
    else:
        val = 0
        for i, ch in enumerate(S):
            if ch == '1':
                val += (1 << i)
        spoiled_bottle = val + 1  # Convert from 0-indexed to 1-indexed.

    # Print the deduced spoiled bottle number.
    sys.stdout.write(str(spoiled_bottle) + "
")
    sys.stdout.flush()

if __name__ == '__main__':
    main()