def main():
	data = input().split()
	M = int(data[0])
	D = int(data[1])
	data2 = input().split()
	y = int(data2[0])
	m = int(data2[1])
	d = int(data2[2])
	
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
			
	print(y_next, m_next, d_next)

if __name__ == "__main__":
	main()