function [confusion, recall, precision, specificity, F1, false_alarm_rate ] = evaluationMetrics(labels,classifications)
    
    TP = 0; FP = 0; TN = 0; FN = 0;

    for i = 1:size(labels,1)
        
        label = labels(i,1);
        
        %Positive Prediction (P) is 1 
        if classifications(i,1) == 1
            if label == 1
                TP = TP + 1;
            else
                FP = FP + 1;
            end
        %Negative Prediction (N) is -1   
        else
            if label == 1
                FN = FN + 1;
            else
                TN = TN + 1;
            end
        end    
    end  
    
    confusion = [TP TN FP FN];
    recall = TP / (TP+FN); %hit rate
    precision = TP / (TP+FP);
    specificity = TN/(TN+FP);
    F1 = (2*TP)/((2*TP)+FN+FP); %F-measure
    false_alarm_rate = FP /(FP+TN);
    
end

