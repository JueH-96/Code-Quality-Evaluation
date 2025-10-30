import math

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    k = int(data[1])
    
    if n == 1:
        print('1 ' * k)
        return
    
    counts = [k] * n
    a = 1
    b = 2
    result = []
    
    while len(result) < n * k:
        sum_old = sum(counts)
        sum_so_far = 0
        selected = -1
        sum_before_x = 0
        
        for x in range(n):
            current_count = counts[x]
            next_sum_so_far = sum_so_far + current_count
            if next_sum_so_far * b >= a * sum_old:
                selected = x + 1
                sum_before_x = sum_so_far
                break
            else:
                sum_so_far = next_sum_so_far
        
        new_a = a * sum_old - b * sum_before_x
        new_b = b * counts[selected - 1]
        g = math.gcd(new_a, new_b)
        if g != 0:
            new_a //= g
            new_b //= g
        a, b = new_a, new_b
        
        counts[selected - 1] -= 1
        result.append(str(selected))
    
    print(' '.join(result))

if __name__ == "__main__":
    main()