def main():
    data = input().split()
    R = int(data[0])
    G = int(data[1])
    B = int(data[2])
    C = input().strip()
    
    if C == "Red":
        result = min(G, B)
    elif C == "Green":
        result = min(R, B)
    elif C == "Blue":
        result = min(R, G)
    else:
        result = min(R, G, B)
    
    print(result)

if __name__ == "__main__":
    main()