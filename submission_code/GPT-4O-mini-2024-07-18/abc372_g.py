def count_valid_pairs(test_cases):
    results = []
    for case in test_cases:
        N, sequences = case
        count = 0
        
        for A, B, C in sequences:
            # We need to find pairs (x, y) such that A*x + B*y < C
            # Rearranging gives us y < (C - A*x) / B
            # y must be a positive integer, hence we need:
            # 1 <= y < (C - A*x) / B
            # This means (C - A*x) must be positive for y to have valid values
            
            # We will iterate over possible values of x
            x = 1
            while True:
                # Calculate the maximum y for this x
                max_y = (C - A * x) // B
                if max_y < 1:
                    break  # No valid y for this x
                count += max_y
                x += 1
        
        results.append(count)
    
    return results

import sys
input = sys.stdin.read

def main():
    data = input().splitlines()
    T = int(data[0])
    index = 1
    test_cases = []
    
    for _ in range(T):
        N = int(data[index])
        sequences = []
        for i in range(N):
            A, B, C = map(int, data[index + 1 + i].split())
            sequences.append((A, B, C))
        test_cases.append((N, sequences))
        index += N + 1
    
    results = count_valid_pairs(test_cases)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()