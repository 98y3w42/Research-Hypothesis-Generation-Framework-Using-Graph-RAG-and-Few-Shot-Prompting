import os
import json
import statistics
import csv

def calculate_averages():
    # カレントディレクトリ内のファイルを取得
    files = [f for f in os.listdir('./data') if f.startswith('evld') and f.endswith('.json')]

    # 各ファイルについて平均値を計算
    # 出力CSVファイルを作成
    output_file = './data/output.csv'
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Matched Score", "Cosine Similarity"])
        macthed_score = list(range(6))
        cosine_simi = []
        for file in files:
            print(file)
            with open('./data/' + file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if type(data[0]) is list:
                scores = []
                cosines = []
                for temp in data:
                    scores.extend([entry['score'] for entry in temp if 'score' in entry])
                    cosines.extend([entry['cosine'] for entry in temp if 'cosine' in entry])
                
                for i in range(len(scores)):
                    csvwriter.writerow([scores[i], cosines[i]])
                #csvwriter.writerow([file, "score"] + scores)
                #csvwriter.writerow([file, "cosine"] + cosines)
                # 平均を計算
                average_score = sum(scores) / len(scores) if scores else 0
                average_cosine = sum(cosines) / len(cosines) if cosines else 0
                    
                # 結果を出力
                print(f"{file}, average_score: {average_score:.2f}, average_cosine: {average_cosine:.3f}")
                    
                pvariance_score = statistics.pvariance(scores)
                pvariance_cosine = statistics.pvariance(cosines)
                    
                print(f"{file}, pvariance_score: {pvariance_score:.2f}, pvariance_cosine: {pvariance_cosine:.3f}")
                for score in scores:
                    macthed_score[score] += 1
                for cosine in cosines:
                    cosine_simi.append(cosine)
                continue

            # `score`と`cosine`のリストを取得
            scores = [entry['score'] for entry in data if 'score' in entry]
            cosines = [entry['cosine'] for entry in data if 'cosine' in entry]
            
            #csvwriter.writerow([file, "score"] + scores)
            #csvwriter.writerow([file, "cosine"] + cosines)
            
            for i in range(len(scores)):
                csvwriter.writerow([scores[i], cosines[i]])

            # 平均を計算
            average_score = sum(scores) / len(scores) if scores else 0
            average_cosine = sum(cosines) / len(cosines) if cosines else 0
            
            # 結果を出力
            print(f"{file}, average_score: {average_score:.2f}, average_cosine: {average_cosine:.3f}")
            
            pvariance_score = statistics.pvariance(scores)
            pvariance_cosine = statistics.pvariance(cosines)
            
            print(f"{file}, pvariance_score: {pvariance_score:.2f}, pvariance_cosine: {pvariance_cosine:.3f}")
            for score in scores:
                macthed_score[score] += 1
            for cosine in cosines:
                cosine_simi.append(cosine)
        print(macthed_score)
        print(sum(macthed_score))
        average_cosine = sum(cosine_simi) / len(cosine_simi) if cosine_simi else 0
        pvariance_cosine = statistics.pvariance(cosine_simi)
        print(f"average_cosine: {average_cosine:.2f}, pvariance_cosine: {pvariance_cosine:.3f}")
            

if __name__ == "__main__":
    calculate_averages()