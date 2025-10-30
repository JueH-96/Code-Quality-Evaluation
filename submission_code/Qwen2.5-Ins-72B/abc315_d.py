# YOUR CODE HERE
def mark_and_remove(cookies, H, W):
    marked = [[False] * W for _ in range(H)]
    
    # Mark rows
    for i in range(H):
        if all(cookies[i][j] == cookies[i][0] for j in range(W)):
            for j in range(W):
                marked[i][j] = True
    
    # Mark columns
    for j in range(W):
        if all(cookies[i][j] == cookies[0][j] for i in range(H)):
            for i in range(H):
                marked[i][j] = True
    
    # Remove marked cookies
    for i in range(H):
        for j in range(W):
            if marked[i][j]:
                cookies[i][j] = '.'
    
    return cookies

def count_remaining(cookies, H, W):
    return sum(1 for i in range(H) for j in range(W) if cookies[i][j] != '.')

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H, W = map(int, data[0].split())
    cookies = data[1:H+1]
    
    while True:
        new_cookies = mark_and_remove(cookies, H, W)
        if new_cookies == cookies:
            break
        cookies = new_cookies
    
    print(count_remaining(cookies, H, W))

if __name__ == "__main__":
    main()