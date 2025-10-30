import sys
import math

def min_operations(test_cases):
    results = []
    for case in test_cases:
        n, k, a = case
        if k == 2:
            # Check if any element is even
            has_even = any(x % 2 == 0 for x in a)
            if has_even:
                results.append(0)
            else:
                # Find the smallest increment to make an element even
                min_ops = float('inf')
                for x in a:
                    if x % 2 != 0:
                        ops = 1
                        min_ops = min(min_ops, ops)
                results.append(min_ops)
        elif k == 3:
            # Check if any element is divisible by 3
            has_divisible = any(x % 3 == 0 for x in a)
            if has_divisible:
                results.append(0)
            else:
                # Find the smallest increment to make an element divisible by 3
                min_ops = float('inf')
                for x in a:
                    if x % 3 != 0:
                        ops = (3 - (x % 3)) % 3
                        min_ops = min(min_ops, ops)
                results.append(min_ops)
        elif k == 4:
            # Count the number of even elements
            even_count = sum(1 for x in a if x % 2 == 0)
            if even_count >= 2:
                results.append(0)
            else:
                # Find the minimal operations to make at least two elements even
                # Option 1: Make one element even and another even
                # Option 2: Make one element divisible by 4
                min_ops = float('inf')
                # Option 1: Make two elements even
                # Find the two elements that can be made even with minimal operations
                # Since even_count < 2, we need to make at least one element even
                # So, find the minimal operations to make one element even
                # Then, find the minimal operations to make another element even
                # But since we need to make at least two elements even, we need to find the two elements that can be made even with the least total operations
                # So, find the two elements with the smallest (x % 2) == 1, and make them even
                # So, for each element, calculate the operations needed to make it even
                # Then, select the two with the smallest operations
                even_ops = []
                for x in a:
                    if x % 2 != 0:
                        even_ops.append(1)
                    else:
                        even_ops.append(0)
                # Sort the even_ops and take the two smallest
                even_ops_sorted = sorted(even_ops)
                if len(even_ops_sorted) >= 2:
                    min_ops_option1 = even_ops_sorted[0] + even_ops_sorted[1]
                else:
                    min_ops_option1 = float('inf')
                # Option 2: Make one element divisible by 4
                # Find the element that can be made divisible by 4 with minimal operations
                min_ops_option2 = float('inf')
                for x in a:
                    if x % 4 != 0:
                        ops = (4 - (x % 4)) % 4
                        min_ops_option2 = min(min_ops_option2, ops)
                # Choose the minimum of the two options
                min_ops = min(min_ops_option1, min_ops_option2)
                results.append(min_ops)
        elif k == 5:
            # Check if any element is divisible by 5
            has_divisible = any(x % 5 == 0 for x in a)
            if has_divisible:
                results.append(0)
            else:
                # Find the smallest increment to make an element divisible by 5
                min_ops = float('inf')
                for x in a:
                    if x % 5 != 0:
                        ops = (5 - (x % 5)) % 5
                        min_ops = min(min_ops, ops)
                results.append(min_ops)
    return results

def main():
    input = sys.stdin.read().split()
    ptr = 0
    t = int(input[ptr])
    ptr += 1
    test_cases = []
    for _ in range(t):
        n = int(input[ptr])
        k = int(input[ptr+1])
        ptr += 2
        a = list(map(int, input[ptr:ptr+n]))
        ptr += n
        test_cases.append((n, k, a))
    results = min_operations(test_cases)
    for res in results:
        print(res)

if __name__ == "__main__":
    main()