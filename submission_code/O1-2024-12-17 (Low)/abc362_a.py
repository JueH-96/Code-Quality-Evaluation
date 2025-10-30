def main():
    import sys
    data = sys.stdin.read().strip().split()
    R, G, B = map(int, data[:3])
    C = data[3]

    prices = []
    if C != "Red":
        prices.append(R)
    if C != "Green":
        prices.append(G)
    if C != "Blue":
        prices.append(B)

    print(min(prices))

# Do not remove the below function call
main()