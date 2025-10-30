n = int(input())
if n == 0:
    print(0)
else:
    a = list(map(int, input().split()))
    
    max_length = 0
    
    # Check all possible starting positions
    for start in range(n):
        count = {}
        
        # Try extending from this start position
        for end in range(start, n):
            # Add current element to count
            num = a[end]
            count[num] = count.get(num, 0) + 1
            
            length = end - start + 1
            
            # Only check even lengths
            if length % 2 == 0:
                # Check if it forms valid pairs
                valid_pairs = True
                for i in range(start, end + 1, 2):
                    if a[i] != a[i + 1]:
                        valid_pairs = False
                        break
                
                if valid_pairs:
                    # Check if all numbers appear exactly twice
                    valid_count = True
                    for cnt in count.values():
                        if cnt != 2:
                            valid_count = False
                            break
                    
                    if valid_count:
                        max_length = max(max_length, length)
            
            # If any number appears more than twice, no point continuing
            if any(cnt > 2 for cnt in count.values()):
                break
    
    print(max_length)