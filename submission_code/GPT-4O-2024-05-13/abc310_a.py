# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    P = int(data[1])
    Q = int(data[2])
    D = list(map(int, data[3:]))
    
    min_dish_price = min(D)
    min_total_with_coupon = Q + min_dish_price
    min_total_without_coupon = P
    
    print(min(min_total_with_coupon, min_total_without_coupon))

if __name__ == "__main__":
    main()