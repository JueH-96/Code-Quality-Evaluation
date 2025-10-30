import numpy as np

MOD = 998244353

class Node:
    def __init__(self, mask, isBoundary, value):
        self.mask = mask
        self.isBoundary = isBoundary
        self.value = value

    def isValid(self, lastVal):
        return (self.isBoundary or lastVal != self.value) and (1 << self.value not in self.mask)

    def __hash__(self):
        return hash((self.mask, self.isBoundary, self.value))

    def __eq__(self, other):
        return self.mask == other.mask and self.isBoundary == other.isBoundary and self.value == other.value

def getInitialNodes(k):
    initial = [Node(0, False, i) for i in range(4)]
    for i in range(1, k):
        nodes = []
        for node in initial:
            for value in range(4):
                if node.isValid(value):
                    newNode = Node((1 << node.value) | node.mask, False, value)
                    nodes.append(newNode)
        initial = nodes
    return initial

def getDP(h, w, k, initialNodeList):
    dp = np.zeros((h + 1, len(initialNodeList)), np.int64)
    dp[0, 0] = 1
    for i in range(h):
        for j in range(w):
            isBoundary = (j == w - 1)
            for nodeIndex, node in enumerate(initialNodeList):
                for value in range(4):
                    if node.isValid(value):
                        nextNode = Node(node.mask, isBoundary, value)
                        nextIndex = initialNodeList.index(nextNode)
                        dp[i + 1, nextIndex] += dp[i, nodeIndex]
                        dp[i + 1, nextIndex] %= MOD
    return dp

def main():
    h, w = map(int, input().split())
    a = [input() for _ in range(h)]

    k = max(int(np.log2(sum(1 for aij in a for c in aij if c == '?'))) * 4, 3)
    k = min(18, k)  # limit for memory and time constraints
    initialNodeList = getInitialNodes(k)

    dp = getDP(h * w, 4, k, initialNodeList)
    answer = dp[h * w - 1].sum() % MOD

    for i in range(h):
        for j in range(w):
            if a[i][j] != '?':
                node = Node(0, False, int(a[i][j]) - 1)
                index = initialNodeList.index(node)
                answer -= dp[i * w + j, index]
                answer %= MOD

    print(answer)

main()