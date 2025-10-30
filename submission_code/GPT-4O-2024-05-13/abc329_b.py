# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    max_value = max(A)
    second_max = -1
    
    for num in A:
        if num != max_value and num > second_max:
            second_max = num
    
    print(second_max)

if __name__ == "__main__":
    main()