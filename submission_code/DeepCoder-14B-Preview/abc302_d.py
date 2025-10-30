import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx +=1
    M = int(input[idx]); idx +=1
    D = int(input[idx]); idx +=1
    
    A = list(map(int, input[idx:idx+N]))
    idx += N
    B = list(map(int, input[idx:idx+M]))
    
    B_sorted = sorted(B)
    
    max_sum = -1
    
    for a in A:
        upper = a + D
        high = bisect.bisect_right(B_sorted, upper)
        if high == 0:
            continue
        else:
            candidate = B_sorted[high - 1]
            if candidate >= a - D:
                current_sum = a + candidate
                if current_sum > max_sum:
                    max_sum = current_sum
    
    print(max_sum if max_sum != -1 else -1)

if __name__ == '__main__':
    main()