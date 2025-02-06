library(lme4)     # For GLMMs
library(dplyr)    # For data manipulation
library(ggplot2)  # For plotting
library(readxl)   # For reading Excel files
library(lmerTest) # p-values (not part of the standard lme4 packages)


# ---- DATA ----

data <- read_excel("data/processed/lmm_data.xlsx")
data$drug <- as.factor(data$drug)
data$time <- as.factor(data$time)


# ---- LINEAR MIXED-EFFECTS MODELS ----

cat("\n\n---------------- DELTA ----------------\n\n")

cat("\n\n- Effect of the interaction\n\n")
model_delta <- lmer(delta ~ drug + time + drug:time + (1 | id), data = data)
print(summary(model_delta))

# model_delta_reduced <- lmer(delta ~ drug + time + (1 | id), data = data)                      
# anova_result <- anova(model_delta_reduced, model_delta, test = "LRT")
# print(anova_result)

# cat("\n\n- Effect of time\n\n")
# model_delta <- lmer(delta ~ drug + time + drug:time + (1 | id), data = data)
# model_delta_reduced <- lmer(delta ~ drug + (1 | id), data = data)                      
# anova_result <- anova(model_delta_reduced, model_delta, test = "LRT")
# print(anova_result)

# cat("\n\n- Effect of drug\n\n")
# model_delta <- lmer(delta ~ drug + time + drug:time + (1 | id), data = data)
# model_delta_reduced <- lmer(delta ~ time + (1 | id), data = data)                      
# anova_result <- anova(model_delta_reduced, model_delta, test = "LRT")
# print(anova_result)


cat("\n\n---------------- THETA ----------------\n\n")

cat("\n\n- Effect of the interaction\n\n")
model_theta <- lmer(theta ~ drug + time + drug:time + (1 | id), data = data)
print(summary(model_theta))

# model_theta_reduced <- lmer(theta ~ drug + time + (1 | id), data = data)                      
# anova_result <- anova(model_theta_reduced, model_theta, test = "LRT")
# print(anova_result)

# cat("\n\n- Effect of time\n\n")
# model_theta <- lmer(theta ~ drug + time + drug:time + (1 | id), data = data)
# model_theta_reduced <- lmer(theta ~ drug + (1 | id), data = data)                      
# anova_result <- anova(model_theta_reduced, model_theta, test = "LRT")
# print(anova_result)

# cat("\n\n- Effect of drug\n\n")
# model_theta <- lmer(theta ~ drug + time + drug:time + (1 | id), data = data)
# model_theta_reduced <- lmer(theta ~ time + (1 | id), data = data)                      
# anova_result <- anova(model_theta_reduced, model_theta, test = "LRT")
# print(anova_result)


# cat("\n\n---------------- ALPHA ----------------\n\n")

# cat("\n\n- Effect of the interaction\n\n")
# model_alpha <- lmer(alpha ~ drug + time + drug:time + (1 | id), data = data)
# model_alpha_reduced <- lmer(alpha ~ drug + time + (1 | id), data = data)                      
# anova_result <- anova(model_alpha_reduced, model_alpha, test = "LRT")
# print(anova_result)

# cat("\n\n- Effect of time\n\n")
# model_alpha <- lmer(alpha ~ drug + time + drug:time + (1 | id), data = data)
# model_alpha_reduced <- lmer(alpha ~ drug + (1 | id), data = data)                      
# anova_result <- anova(model_alpha_reduced, model_alpha, test = "LRT")
# print(anova_result)

# cat("\n\n- Effect of drug\n\n")
# model_alpha <- lmer(alpha ~ drug + time + drug:time + (1 | id), data = data)
# model_alpha_reduced <- lmer(alpha ~ time + (1 | id), data = data)                      
# anova_result <- anova(model_alpha_reduced, model_alpha, test = "LRT")
# print(anova_result)


# cat("\n\n---------------- RATIO ----------------\n\n")

# cat("\n\n- Effect of the interaction\n\n")
# model_ratio <- lmer(ratio ~ drug + time + drug:time + (1 | id), data = data)
# model_ratio_reduced <- lmer(ratio ~ drug + time + (1 | id), data = data)                      
# anova_result <- anova(model_ratio_reduced, model_ratio, test = "LRT")
# print(anova_result)

# cat("\n\n- Effect of time\n\n")
# model_ratio <- lmer(ratio ~ drug + time + drug:time + (1 | id), data = data)
# model_ratio_reduced <- lmer(ratio ~ drug + (1 | id), data = data)                      
# anova_result <- anova(model_ratio_reduced, model_ratio, test = "LRT")
# print(anova_result)

# cat("\n\n- Effect of drug\n\n")
# model_ratio <- lmer(ratio ~ drug + time + drug:time + (1 | id), data = data)
# model_ratio_reduced <- lmer(ratio ~ time + (1 | id), data = data)                      
# anova_result <- anova(model_ratio_reduced, model_ratio, test = "LRT")
# print(anova_result)


# cat("\n\n---------------- PE ----------------\n\n")

# cat("\n\n- Effect of the interaction\n\n")
# model_pe <- lmer(pe ~ drug + time + drug:time + (1 | id), data = data)
# model_pe_reduced <- lmer(pe ~ drug + time + (1 | id), data = data)                      
# anova_result <- anova(model_pe_reduced, model_pe, test = "LRT")
# print(anova_result)

# cat("\n\n- Effect of time\n\n")
# model_pe <- lmer(pe ~ drug + time + drug:time + (1 | id), data = data)
# model_pe_reduced <- lmer(pe ~ drug + (1 | id), data = data)                      
# anova_result <- anova(model_pe_reduced, model_pe, test = "LRT")
# print(anova_result)

# cat("\n\n- Effect of drug\n\n")
# model_pe <- lmer(pe ~ drug + time + drug:time + (1 | id), data = data)
# model_pe_reduced <- lmer(pe ~ time + (1 | id), data = data)                      
# anova_result <- anova(model_pe_reduced, model_pe, test = "LRT")
# print(anova_result)


# cat("\n\n---------------- SE ----------------\n\n")

# cat("\n\n- Effect of the interaction\n\n")
# model_se <- lmer(se ~ drug + time + drug:time + (1 | id), data = data)
# model_se_reduced <- lmer(se ~ drug + time + (1 | id), data = data)                      
# anova_result <- anova(model_se_reduced, model_se, test = "LRT")
# print(anova_result)

# cat("\n\n- Effect of time\n\n")
# model_se <- lmer(se ~ drug + time + drug:time + (1 | id), data = data)
# model_se_reduced <- lmer(se ~ drug + (1 | id), data = data)                      
# anova_result <- anova(model_se_reduced, model_se, test = "LRT")
# print(anova_result)

# cat("\n\n- Effect of drug\n\n")
# model_se <- lmer(se ~ drug + time + drug:time + (1 | id), data = data)
# model_se_reduced <- lmer(se ~ time + (1 | id), data = data)                      
# anova_result <- anova(model_se_reduced, model_se, test = "LRT")
# print(anova_result)


cat("\n\n---------------- FNIRS ----------------\n\n")

cat("\n\n- Effect of the interaction\n\n")
model_fnirs_1 <- lmer(fnirs_1 ~ drug + time + drug:time + (1 | id), data = data)
print(summary(model_fnirs_1))

# model_fnirs_1_reduced <- lmer(fnirs_1 ~ drug + time + (1 | id), data = data)                      
# anova_result <- anova(model_fnirs_1_reduced, model_fnirs_1, test = "LRT")
# print(anova_result)

# cat("\n\n- Effect of time\n\n")
# model_fnirs_1 <- lmer(fnirs_1 ~ drug + time + drug:time + (1 | id), data = data)
# model_fnirs_1_reduced <- lmer(fnirs_1 ~ drug + (1 | id), data = data)                      
# anova_result <- anova(model_fnirs_1_reduced, model_fnirs_1, test = "LRT")
# print(anova_result)

# cat("\n\n- Effect of drug\n\n")
# model_fnirs_1 <- lmer(fnirs_1 ~ drug + time + drug:time + (1 | id), data = data)
# model_fnirs_1_reduced <- lmer(fnirs_1 ~ time + (1 | id), data = data)                      
# anova_result <- anova(model_fnirs_1_reduced, model_fnirs_1, test = "LRT")
# print(anova_result)


# cat("\n\n---------------- PUPILLOMETRY ----------------\n\n")

# cat("\n\n- Effect of the interaction\n\n")
# model_pupillometry_score <- lmer(pupillometry_score ~ drug + time + drug:time + (1 | id), data = data)
# model_pupillometry_score_reduced <- lmer(pupillometry_score ~ drug + time + (1 | id), data = data)                      
# anova_result <- anova(model_pupillometry_score_reduced, model_pupillometry_score, test = "LRT")
# print(anova_result)

# cat("\n\n- Effect of time\n\n")
# model_pupillometry_score <- lmer(pupillometry_score ~ drug + time + drug:time + (1 | id), data = data)
# model_pupillometry_score_reduced <- lmer(pupillometry_score ~ drug + (1 | id), data = data)                      
# anova_result <- anova(model_pupillometry_score_reduced, model_pupillometry_score, test = "LRT")
# print(anova_result)

# cat("\n\n- Effect of drug\n\n")
# model_pupillometry_score <- lmer(pupillometry_score ~ drug + time + drug:time + (1 | id), data = data)
# model_pupillometry_score_reduced <- lmer(pupillometry_score ~ time + (1 | id), data = data)                      
# anova_result <- anova(model_pupillometry_score_reduced, model_pupillometry_score, test = "LRT")
# print(anova_result)

