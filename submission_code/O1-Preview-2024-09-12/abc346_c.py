# YOUR CODE HERE
import sys

def main():
    import threading
    def run():
        input_data = sys.stdin.read().split()
        N = int(input_data[0])
        K = int(input_data[1])
        A = map(int, input_data[2:])
        s = set()
        for a in A:
            if 1 <= a <= K:
                s.add(a)
        total_sum = K * (K + 1) // 2
        sum_s = sum(s)
        print(total_sum - sum_s)
    threading.Thread(target=run).start()

if __name__ == "__main__":
    main()