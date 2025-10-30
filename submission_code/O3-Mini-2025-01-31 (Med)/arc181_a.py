def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    t = int(data[0])
    pos = 1
    answers = []
    
    for _ in range(t):
        n = int(data[pos])
        pos += 1
        # Read permutation P (1-indexed logically, but stored as 0-indexed)
        P = list(map(int, data[pos:pos+n]))
        pos += n
        
        # If the permutation is already sorted (i.e. identity), answer is 0.
        already_sorted = True
        for i in range(n):
            if P[i] != i+1:
                already_sorted = False
                break
        if already_sorted:
            answers.append("0")
            continue
        
        # Precompute prefix maximum and suffix minimum arrays.
        # prefix_max[i]: maximum value in P[0..i]
        prefix_max = [0] * n
        prefix_max[0] = P[0]
        for i in range(1, n):
            prefix_max[i] = P[i] if P[i] > prefix_max[i-1] else prefix_max[i-1]
        
        # suffix_min[i]: minimum value in P[i..n-1]
        suffix_min = [0] * n
        suffix_min[n-1] = P[n-1]
        for i in range(n-2, -1, -1):
            suffix_min[i] = P[i] if P[i] < suffix_min[i+1] else suffix_min[i+1]
        
        # We try to check if one operation can do the job.
        # Operation: choose an index k (1-indexed) such that:
        # 1. P[k] = k,
        # 2. The left segment P[1..k-1] (if exists) when sorted becomes [1, 2, ..., k-1].
        #    In a permutation, the necessary and sufficient condition is that the maximum in that segment is k-1.
        # 3. The right segment P[k+1..N] (if exists) when sorted becomes [k+1, k+2, ..., N].
        #    This is equivalent to the minimum in that segment being k+1.
        # If such a k exists then one operation (by choosing that k) will sort both segments
        # while leaving P[k] intact. So the answer is 1.
        one_operation_possible = False
        for k in range(1, n+1):
            if P[k-1] != k:
                continue  # The element at position k must be k.
                
            # Check left segment if exists.
            left_ok = True
            if k > 1:
                # Since the left part is a permutation of its elements,
                # if the maximum is k-1 then it must contain {1,2,...,k-1}.
                if prefix_max[k-2] != k-1:
                    left_ok = False
            
            # Check right segment if exists.
            right_ok = True
            if k < n:
                # Similarly, if the minimum is k+1 then the segment must be exactly {k+1, k+2, ..., n}.
                if suffix_min[k] != k+1:
                    right_ok = False
            
            if left_ok and right_ok:
                one_operation_possible = True
                break
        
        if one_operation_possible:
            answers.append("1")
        else:
            answers.append("2")
    
    sys.stdout.write("
".join(answers))

if __name__ == '__main__':
    main()