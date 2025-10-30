# YOUR CODE HERE
import sys
def main():
    import threading
    def run():
        N = int(sys.stdin.readline())
        cards = []
        for i in range(1, N+1):
            A_i, C_i = map(int, sys.stdin.readline().split())
            cards.append( (-A_i, C_i, i) )

        cards.sort()

        min_C = float('inf')
        remaining_indices = []
        for neg_A_i, C_i, index in cards:
            if C_i > min_C:
                # Discarded
                continue
            else:
                min_C = C_i
                remaining_indices.append(index)
        remaining_indices.sort()
        print(len(remaining_indices))
        print(' '.join(map(str, remaining_indices)))
    threading.Thread(target=run).start()
main()