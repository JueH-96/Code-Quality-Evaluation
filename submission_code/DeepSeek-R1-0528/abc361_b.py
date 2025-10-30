def main():
    data1 = list(map(int, input().split()))
    data2 = list(map(int, input().split()))
    
    a, b, c, d, e, f = data1
    g, h, i, j, k, l = data2
    
    if max(a, g) < min(d, j) and max(b, h) < min(e, k) and max(c, i) < min(f, l):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()