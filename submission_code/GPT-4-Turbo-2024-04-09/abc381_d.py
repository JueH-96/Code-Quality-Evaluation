def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    if N == 0:
        print(0)
        return
    
    max_length = 0
    i = 0
    
    while i < N:
        # Start a potential 1122 sequence
        count = {}
        valid = True
        j = i
        
        while j < N and valid:
            num = A[j]
            if num in count:
                count[num] += 1
            else:
                count[num] = 1
            
            # Check if the current number has appeared more than twice
            if count[num] > 2:
                valid = False
            elif count[num] == 2:
                # Check if the sequence can still be a 1122 sequence
                # We need pairs to be consecutive and the same
                if j % 2 == 1 and A[j] != A[j-1]:
                    valid = False
            
            if valid and j % 2 == 1:  # Only consider even-length segments
                # Check if all numbers have appeared exactly twice
                all_twice = all(c == 2 for c in count.values())
                if all_twice:
                    max_length = max(max_length, j - i + 1)
            
            j += 1
        
        # Move to the next starting point
        i += 1
    
    print(max_length)

if __name__ == "__main__":
    main()