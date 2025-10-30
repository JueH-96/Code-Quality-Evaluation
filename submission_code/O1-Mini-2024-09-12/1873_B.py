# YOUR CODE HERE
import sys
import math

def main():
    input = sys.stdin.read().split()
    idx = 0
    t = int(input[idx])
    idx +=1
    for _ in range(t):
        if idx >= len(input):
            break
        n = int(input[idx])
        idx +=1
        a = list(map(int, input[idx:idx+n]))
        idx +=n
        max_product = 0
        for i in range(n):
            modified = a.copy()
            modified[i] +=1
            product = 1
            for num in modified:
                product *= num
            if product > max_product:
                max_product = product
        print(max_product)

if __name__ == "__main__":
    main()