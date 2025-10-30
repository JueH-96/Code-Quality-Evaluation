import sys

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    X = int(input[idx])
    idx +=1
    
    processes = []
    for _ in range(N):
        A = int(input[idx])
        P = int(input[idx+1])
        B = int(input[idx+2])
        Q = int(input[idx+3])
        processes.append( (A, P, B, Q) )
        idx +=4
    
    def is_possible(m):
        total_cost = 0
        for A, P, B, Q in processes:
            min_cost = float('inf')
            # Check all possible s up to B
            max_s = B
            for s in range(0, max_s +1):
                a = A * s
                if a >= m:
                    t = 0
                else:
                    remaining = m - a
                    t = (remaining + B -1) // B
                current_cost = s * P + t * Q
                if current_cost < min_cost:
                    min_cost = current_cost
            # Check all possible t up to A
            max_t = A
            for t in range(0, max_t +1):
                b = B * t
                if b >= m:
                    s = 0
                else:
                    remaining = m - b
                    s = (remaining + A -1) // A
                current_cost = s * P + t * Q
                if current_cost < min_cost:
                    min_cost = current_cost
            if min_cost == float('inf'):
                return False
            total_cost += min_cost
            if total_cost > X:
                return False
        return True
    
    # Binary search
    left = 0
    right = 10**18
    answer = 0
    while left <= right:
        mid = (left + right) // 2
        if mid == 0:
            # Special case, all processes can have zero cost
            ok = True
        else:
            ok = is_possible(mid)
        if ok:
            answer = mid
            left = mid +1
        else:
            right = mid -1
    
    print(answer)

if __name__ == "__main__":
    main()