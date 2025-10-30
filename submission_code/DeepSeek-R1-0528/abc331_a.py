def main():
    data = input().split()
    M = int(data[0])
    D = int(data[1])
    date_line = input().split()
    y = int(date_line[0])
    m = int(date_line[1])
    d = int(date_line[2])
    
    if d < D:
        d_next = d + 1
        m_next = m
        y_next = y
    else:
        d_next = 1
        if m < M:
            m_next = m + 1
            y_next = y
        else:
            m_next = 1
            y_next = y + 1
            
    print(f"{y_next} {m_next} {d_next}")

if __name__ == "__main__":
    main()