# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    input_data = sys.stdin.read().split()
    n = int(input_data[0])
    A = []
    B = []
    total_A = 0
    max_B_minus_A = 0
    idx = 1
    for _ in range(n):
        a = int(input_data[idx])
        b = int(input_data[idx+1])
        total_A += a
        if b - a > max_B_minus_A:
            max_B_minus_A = b - a
        idx += 2
    result = total_A + max_B_minus_A
    print(result)

threading.Thread(target=main).start()