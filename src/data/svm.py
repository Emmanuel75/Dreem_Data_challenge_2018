def wake_svc(X, y, X_test, y_true):
    Lkappa_l1 = []
    Lkappa_l2 = []
    Lsparsity_l1 = []
    Lsparsity_l2 = []
    Lrecall_l1 = []
    Lrecall_l2 = []
    Lprecision_l1 = []
    Lprecision_l2 = []
    L = [0.00000001,0.0000001, 0.000001, 0.00001, 0.0001, 0.01, 1, 100]
    for C in L: 
        l1_Lsvc = LinearSVC(C=C, penalty='l1', loss="squared_hinge")
        l2_Lsvc = LinearSVC(C=C, penalty='l2', loss="hinge")
        l1_Lsvc.fit(X, y)
        l2_Lsvc.fit(X, y)
        y_pred_l1 = l1_Lsvc.predict(X_test)
        y_pred_l2 = Ll2_svc.predict(X_test)
        coef_l1_Lsvc = l1_Lsvc.coef_.ravel()
        coef_l2_Lsvc = l2_Lsvc.coef_.ravel()
        sparsity_l1_Lsvc = np.mean(coef_l1_Lsvc == 0) * 100
        sparsity_l2_Lsvc = np.mean(coef_l2_Lsvc == 0) * 100
        kappa_l1 = cohen_kappa_score(y_true, y_pred_l1)
        kappa_l2 = cohen_kappa_score(y_true, y_pred_l2)
        #
        Lsparsity_l1.append(sparsity_l1_Lsvc)
        Lsparsity_l2.append(sparsity_l2_Lsvc)
        Lrecall_l1.append(recall_score(y_true, y_pred_l1))
        Lrecall_l2.append(recall_score(y_true, y_pred_l2))
        Lprecision_l1.append(precision_score(y_true, y_pred_l1))
        Lprecision_l2.append(precision_score(y_true, y_pred_l2))
        Lkappa_l1.append(kappa_l1)
        Lkappa_l2.append(kappa_l2)
        notScarseCoefs = []
        for i in range(0, len(coef_l1_Lsvc)):
            if (coef_l1_Lsvc[i] !=0):
                notScarseCoefs.append(X.columns[i])
    fig, ax = plt.subplots()
    ax.set_title("Sparsity vs C for l1")
    ax.semilogx(L, Lsparsity_l1)
    plt.show()
    fig, ax = plt.subplots()
    ax.set_title("Sparsity vs C for l2")
    ax.semilogx(L, Lsparsity_l2)
    plt.show()
    fig, ax = plt.subplots()
    ax.set_title("Precision (red) & recall (blue) vs C for l1")
    ax.semilogx(L, Lprecision_l1, color='r')
    ax.semilogx(L, Lrecall_l1, color='b')
    #plt.legend(loc='upper right')
    plt.show()
    fig, ax = plt.subplots()
    ax.set_title("Precision (red) & recall (blue) vs C for l2")
    ax.semilogx(L, Lprecision_l2, color='r')
    ax.semilogx(L, Lrecall_l2, color='b')
    #plt.legend(loc='upper right')
    plt.show()
               
    fig, ax = plt.subplots()
    ax.set_title("kappa vs C for l1")
    ax.semilogx(L, Lkappa_l1)
    #plt.legend(loc='upper right')
    plt.show()
    fig, ax = plt.subplots()
    ax.set_title("kappa vs C for l2")
    ax.semilogx(L, Lkappa_l2)
    plt.show()
    return [Lkappa_l1, Lkappa_l2]
    
    
    
def wake_svc_rbf(X, y, X_test, y_true):
    #Lkappa_l1 = []
    Lkappa = []
    Lrecall = []
    Lprecision = []
    gamma = [0.01, 0.1, 1, 5, 100]
    L = [0.0000001, 0.000001, 0.00001, 0.0001, 0.01, 1, 100]
    for g in gamma:
        for C in L: 
            print("gamma: " + str(g))
            print ("C: " + str(C))
            model_svc =SVC(kernel="rbf", gamma=g, C=C)
            model_svc.fit(X, y)
            y_pred = model_svc.predict(X_test)
            Lrecall.append(recall_score(y_true, y_pred))
            Lprecision.append(precision_score(y_true, y_pred))
            Lkappa.append(cohen_kappa_score(y_true, y_pred))
    ####
        fig, ax = plt.subplots()
        ax.set_title("Precision (red) & recall (blue) vs C with gamma =" + str(g))
        ax.semilogx(L, Lprecision, color='r')
        ax.semilogx(L, Lrecall, color='b')
        plt.show()
                   
        fig, ax = plt.subplots()
        ax.set_title("kappa vs C with gamma ="  + str(g))
        ax.semilogx(L, Lkappa)
        plt.show()
    return Lkappa