스키마 고정(YAML): tables.{pharmacology,pk_adme,tox_repeat,genotox,safety_pharm}에 columns와 rows 필수.

프롬프트: “YAML만 받아 표로 렌더하라. 자유 텍스트 금지.”

렌더러: YAML/CSV→마크다운 표 변환을 코드로 수행

검증: 필수 테이블 존재,열 일치,행 수(≥1) 검사. 실패 시 재생성   ###미정

필요 시 다른 모듈 키(열 이름, 단위, 최소 행 수)를 더 엄격하게 잠글 수 있음



"TM_5_1상.txt"                                  TM_5_Phase1.txt
"TM_5_2상.txt"                                  TM_5_Phase2.txt
"TM_5_3상.txt"                                  TM_5_Phase3.txt
"TM_5_비임상.txt"                               TM_5_Nonclinical.txt
"TM_5_행정_라벨링_KR.txt"                       TM_5_Admin_Labeling_KR.txt
"TM_5_M2_4_비임상개요.txt"                      TM_5_M2_4_Nonclinical_Overview.txt
"TM_5_M2_5_임상개요.txt"                        TM_5_M2_5_Clinical_Overview.txt
"TM_5_M2_6_비임상요약_표_Genotox.csv"           TM_5_M2_6_Nonclinical_Summary_Table_Genotox.csv
"TM_5_M2_6_비임상요약_표_Pharmacology.csv"      TM_5_M2_6_Nonclinical_Summary_Table_Pharmacology.csv
"TM_5_M2_6_비임상요약_표_PKADME.csv"            TM_5_M2_6_Nonclinical_Summary_Table_PKADME.csv
"TM_5_M2_6_비임상요약_표_SafetyPharm.csv"       TM_5_M2_6_Nonclinical_Summary_Table_SafetyPharm.csv
"TM_5_M2_6_비임상요약_표_ToxRepeat.csv"         TM_5_M2_6_Nonclinical_Summary_Table_ToxRepeat.csv
"TM_5_M2_6_비임상요약.yaml"                     TM_5_M2_6_Nonclinical_Summary.yaml
"TM_5_M2_7_노출개요.csv"                        TM_5_M2_7_Exposure_Overview.csv
"TM_5_M2_7_안전성_SOC별.csv"                    TM_5_M2_7_Safety_by_SOC.csv
"TM_5_M2_7_유효성_연구별.csv"                   TM_5_M2_7_Efficacy_by_Study.csv
"TM_5_M2_7_임상요약.yaml"                       M_5_M2_7_Clinical_Summary.yaml
"TM_5_M2_7_중단사유.csv"                        TM_5_M2_7_Discontinuation_Reasons.csv
