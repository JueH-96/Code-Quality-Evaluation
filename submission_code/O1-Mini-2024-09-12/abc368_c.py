# YOUR CODE HERE
import sys

def main():
    import sys

    import sys

    def input():
        return sys.stdin.read()

    data = input().split()
    N = int(data[0])
    H = list(map(int, data[1:N+1]))
    current_T = 0

    for h in H:
        s = current_T +1
        low = 1
        high = h  # maximum k needed
        while low < high:
            mid = (low + high) //2
            # Calculate number of multiples of 3 in [s, s+mid-1]
            multiples = ( (s + mid -1)//3 ) - ( (s -1)//3 )
            damage = multiples*3 + (mid - multiples)*1
            if damage >= h:
                high = mid
            else:
                low = mid +1
        current_T += low

    print(current_T)

if __name__ == "__main__":
    main()