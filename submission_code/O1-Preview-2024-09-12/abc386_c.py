# YOUR CODE HERE
import sys

def isOneEditDistance(S, T):
    if S == T:
        return True  # Zero operations needed
    len_S = len(S)
    len_T = len(T)
    if len_S - len_T > 1 or len_T - len_S > 1:
        return False  # Cannot be transformed with one operation
    if len_S == len_T:
        # Check for one replace operation
        count = 0
        for i in range(len_S):
            if S[i] != T[i]:
                count +=1
                if count >1:
                    return False
        return True
    elif len_S +1 == len_T:
        # Check if inserting one character into S makes it equal to T
        return canInsert(S, T)
    elif len_S == len_T +1:
        # Check if deleting one character from S makes it equal to T
        return canDelete(S, T)
    else:
        return False

def canInsert(S, T):
    # Attempt to insert a character into S to make it equal to T
    indexS = 0
    indexT = 0
    mismatchFound = False
    while indexS < len(S) and indexT < len(T):
        if S[indexS] == T[indexT]:
            indexS +=1
            indexT +=1
        else:
            if mismatchFound:
                return False
            mismatchFound = True
            indexT +=1  # Simulate insertion in S by moving ahead in T
    return True  # All remaining characters match

def canDelete(S, T):
    # Attempt to delete a character from S to make it equal to T
    indexS = 0
    indexT = 0
    mismatchFound = False
    while indexS < len(S) and indexT < len(T):
        if S[indexS] == T[indexT]:
            indexS +=1
            indexT +=1
        else:
            if mismatchFound:
                return False
            mismatchFound = True
            indexS +=1  # Simulate deletion in S by moving ahead in S
    return True  # All remaining characters match

def main():
    import sys
    import threading
    def read_and_solve():
        K = sys.stdin.readline().strip()
        S = sys.stdin.readline().strip()
        T = sys.stdin.readline().strip()
        # Since K=1, we only proceed with K=1
        if isOneEditDistance(S, T):
            print('Yes')
        else:
            print('No')
    threading.Thread(target=read_and_solve).start()

if __name__ == '__main__':
    main()