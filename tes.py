from NNDirectory.LsDirectory.LearningSetCl import LearningSet

ls = LearningSet(path_to_df=r'C:\Users\vgv\Desktop\PythonData\cleanedDf.txt')
my_df = ls.create_learningSet(ls.initial_df)
df_enc = ls.encode_categorials_features(my_df)
df_scale = ls.my_scale(df_enc)

#cor = df_scale.corr()
#plt.matshow(cor)
#plt.xticks(range(len(df_scale.columns)), df_scale.columns)
#plt.yticks(range(len(df_scale.columns)), df_scale.columns)
#plt.colorbar()
#plt.show()