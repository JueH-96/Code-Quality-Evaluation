import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    S = int(data[1])
    K = int(data[2])
    
    total_price = 0
    index = 3
    for _ in range(N):
        P_i = int(data[index])
        Q_i = int(data[index + 1])
        total_price += P_i * Q_i
        index += 2
    
    if total_price >= S:
        shipping_fee = 0
    else:
        shipping_fee = K
    
    total_amount = total_price + shipping_fee
    print(total_amount)

if __name__ == "__main__":
    main()