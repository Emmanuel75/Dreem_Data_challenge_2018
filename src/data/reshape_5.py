def reshape_5(df):
	#df = d.copy()
	if (df.shape[0] <5): 
		return df
	r = df.shape[0]
	c = df.shape[1]
	newColumns = []
	for i in range(0, 5):
		for colName in df.columns:
			newColumns.append(str(colName) + str(i+1))
	result = pd.DataFrame(columns = newColumns, data = np.zeros((r, c*5)))
	for i in range(2, r-2):
		result.iloc[i,0:c]= df.iloc[i-2, :].values
		result.iloc[i,c:c*2]= df.iloc[i-1, :].values
		result.iloc[i,c*2:c*3]= df.iloc[i, :].values
		result.iloc[i,c*3:c*4]= df.iloc[i+1, :].values
		result.iloc[i,c*4:c*5]= df.iloc[i+2, :].values
	
	
	return result
 