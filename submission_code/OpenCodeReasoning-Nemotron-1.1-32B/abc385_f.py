import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	buildings = []
	index = 1
	for i in range(n):
		x = int(data[index])
		h = int(data[index + 1])
		index += 2
		buildings.append((x, h))
	
	visible0 = [True] * n
	for i in range(n):
		M0 = 0.0
		for j in range(i):
			if buildings[j][0] < buildings[i][0]:
				candidate = buildings[j][1] * (buildings[i][0] / buildings[j][0])
				if candidate > M0:
					M0 = candidate
		if M0 >= buildings[i][1]:
			visible0[i] = False
	
	if all(visible0):
		print(-1)
		return
	
	T = [-10**18] * n
	for i in range(n):
		if not visible0[i]:
			for j in range(i):
				if buildings[j][0] < buildings[i][0]:
					num = buildings[j][1] * buildings[i][0] - buildings[i][1] * buildings[j][0]
					den = buildings[i][0] - buildings[j][0]
					t_j = num / den
					if t_j > T[i]:
						T[i] = t_j
	ans = max(T)
	print("{:.15f}".format(ans))

if __name__ == '__main__':
	main()