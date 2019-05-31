
def AnalyzeError(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(20,10))
    plt.subplot(1,2, 1)
    sns.countplot(x=0, data=pd.DataFrame(y_true))
    plt.ylim(0, 4000)
    plt.subplot(1,2, 2)
    sns.countplot(x=0, data=pd.DataFrame(y_pred))
    plt.ylim(0, 4000)
    fig.suptitle("Actual and predicted distribution", size =  'x-large')
    plt.show()
    
    df_ = pd.DataFrame()
    df_["Test"]= y_true
    df_["Pred"] = y_pred
    df_['error'] = df_.Test != df_.Pred
    #sns.countplot(x="Test", data=df_[df_.error])
    
    error0 = df_[(df_.error) & (df_.Test==0)].count()[0] / df_[df_.Test==0].count()[0]
    error1 = df_[(df_.error) & (df_.Test==1)].count()[0] / df_[df_.Test==1].count()[0]
    error2 = df_[(df_.error) & (df_.Test==2)].count()[0] / df_[df_.Test==2].count()[0]
    error3 = df_[(df_.error) & (df_.Test==3)].count()[0] / df_[df_.Test==3].count()[0]
    error4 = df_[(df_.error) & (df_.Test==4)].count()[0] / df_[df_.Test==4].count()[0]

    Lerror = [error0, error1, error2, error3, error4]
    sns.barplot(x=[0, 1, 2, 3, 4], y=Lerror)
    plt.title('Wrongly classified in a phase in percent of the test population for this phase')
    plt.show()