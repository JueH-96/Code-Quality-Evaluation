from scipy.sparse import csr_matrix
from collections import Counter
import numpy as np

def find_remaining_cookies(H, W, cookies):
    """
    Performs a procedure to remove cookies of the same color in rows and columns and
    returns the number of cookies remaining at the end.
    
    Parameters:
    H (int): number of rows
    W (int): number of columns
    cookies (List[str]): a list of strings representing the cookies in the grid
    
    Returns:
    int: number of cookies remaining after the procedure
    """
    m1, m2 = np.zeros((H + W, W), dtype=int), np.zeros((H + W, H), dtype=int)
    c = Counter()
    
    for y in range(H):
        row = cookies[y]
        for x, ch in enumerate(row):
            c[ch] += 1
            m1[y][x] = c[ch]
            m2[y + W][x] = c[ch]
    
    csr1 = csr_matrix(m1)
    csr2 = csr_matrix(m2)
    
    while True:
        any_removed = False
        
        # Remove consecutive equal elements in rows and adjust csr matrices
        for y in range(H):
            row = ""
            e = -1
            for x in range(W):
                if e == m1[y][x]:
                    m1[y][x] = -1
                    csr1.data[csr1.indptr[y]:csr1.indptr[y + 1]][x] = 0
                    any_removed = True
                else:
                    row += cookies[y][x]
                    e = m1[y][x]
            cookies[y] = row
        
        # Remove consecutive equal elements in columns and adjust csr matrices
        for x in range(W):
            col = ""
            e = -1
            for y in range(H):
                if e == m2[y + W][x]:
                    m2[y + W][x] = -1
                    csr2.data[csr2.indptr[y + W]:csr2.indptr[y + W + 1]][x] = 0
                    any_removed = True
                else:
                    col += cookies[y][x] if x < len(cookies[y]) else "-"
                    e = m2[y + W][x]
            c = Counter(col)
            for y in range(H):
                if x < len(cookies[y]) and x < len(cookies):
                    m1[y][x] = c[cookies[y][x]]
                    m2[y + W][x] = c[cookies[y][x]]
        
        if not any_removed:
            break
    
    remaining_count = sum(len(row) for row in cookies)
    return remaining_count

# Example usage
if __name__ == "__main__":
    H, W = map(int, input().split())
    cookies = [input() for _ in range(H)]
    print(find_remaining_cookies(H, W, cookies))