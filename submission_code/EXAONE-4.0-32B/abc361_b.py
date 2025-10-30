def main():
    data1 = list(map(int, input().split()))
    data2 = list(map(int, input().split()))
    
    a, b, c, d, e, f = data1
    g, h, i, j, k, l = data2
    
    overlap_x = max(a, g) < min(d, j)
    overlap_y = max(b, h) < min(e, k)
    overlap_z = max(c, i) < min(f, l)
    
    if overlap_x and overlap_y and overlap_z:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()