#' Copyright (c) 2025. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
#' @Time     : 2025/3/25
#' @Author   : ZL.Z
#' @Email    : xu_jun@mail.ustc.edu.cn
#' @Reference: None
#' @FileName : app.R
#' @Software : R-4.2.0; RStudio; Windows10
#' @Hardware : Intel Core i7-4712MQ; NVIDIA GeForce 840M
#' @Version  : V1.0 - ZL.Z：2025/3/25
#' 		         First version.
#' @License  : None

library(shiny)
library(dplyr)
library(tidyr)
library(ggplot2)
library(survival)
library(rms)
if (!require(survAUC)) {
  install.packages("survAUC")
}
library(randomForestSRC)
library(survAUC)
library(survcomp)
library(readr)

select <- dplyr::select

cur_dir <- getwd()
train_data_f <- file.path(cur_dir, "train_cohort.csv")
load_data <- function(file_path) {
  df <- read_csv(file_path, show_col_types = FALSE)
  x <- df %>% select(Up_to_Seven:Ensemble_DL)
  y <- Surv(time = df$OS_time, event = df$OS_status)
  list(x = x, y = y)
}
train_data <- load_data(train_data_f)
train_df <- cbind(train_data$x, OS_time = train_data$y[,1], OS_status = train_data$y[,2])
train_df$SurvObj <- with(train_df, Surv(OS_time, OS_status))
set.seed(205)
rsf_model <- rfsrc(Surv(OS_time, OS_status) ~ ., data = train_df %>% select(-SurvObj),
                   ntree = 86,
                   nodesize = 2,
                   nsplit = 5,
                   mtry = ceiling(log2(ncol(train_data$x))),
                   importance = TRUE,
                   forest = TRUE,
                   block.size = 1,
                   max.depth = 7)
predict_risk <- function(model, x) {
  predict(model, newdata = x)$predicted
}

ui <- fluidPage(
  tags$head(
    tags$style(HTML("
    body { background-color: black; color: white; font-family: Arial, sans-serif;}
    .well { background-color: #333; border-color: #444; min-height: 600px; }
    .shiny-input-container { color: white; }
    table { color: white; }
    .table th, .table td { color: white; border-color: #666; }
    .shiny-output-error { color: red; }
      .title-panel { 
        background-color: #000; 
        color: #fff; 
        text-align: center; 
        padding: 10px; 
        font-size: 24px; 
        font-weight: bold; 
      }
    "))
  ),
  div(class = "title-panel",
      "MMF System for Personalized Survival Prediction of HCC Immunotherapy",
      windowTitle = "HCC Predictor",
      tags$br(),
      tags$img(src = "logo.png", height = 50)
  ),
  fluidRow(
    column(
      width = 5,
      wellPanel(
        style = "background-color: #333; border-color: #444; min-height: 600px;", # 保持高度一致
        sliderInput("Ensemble_DL", "Ensemble_DL score", min = 0, max = 10, value = 0.01, step = 0.001),
        fluidRow(
          column(6, selectInput("age", "Age (years)", choices = c("<=55", ">55"))),
          column(6, selectInput("Cirrhosis", "Cirrhosis", choices = c("No", "Yes")))
        ),
        fluidRow(
          column(6, selectInput("Child_Pugh", "Child-Pugh", choices = c("A", "B"))),
          column(6, selectInput("AFP_level", "AFP level (ng/ml)", choices = c("<=400", ">400")))
        ),
        fluidRow(
          column(6, selectInput("BCLC_stage", "BCLC stage", choices = c("B", "C"))),
          column(6, selectInput("ECOG_PS", "ECOG PS", choices = c("0", "1")))
        ),
        fluidRow(
          column(6, selectInput("max_tumor_diameter", "Tumor Size(cm)", choices = c("<=10", ">10"))),
          column(6, selectInput("Up_to_Seven", "Up_to_Seven", choices = c("No", "Yes")))
        ),
        fluidRow(
          column(6, selectInput("PVTT", "PVTT", choices = c("0" = "0", "1" = "1", ">=2" = "2"))),
          column(6, selectInput("Line_of_ICIs", "Line of ICIs", choices = c("1" = "1", ">=2" = "2")))
        ),
        fluidRow(
          column(4, selectInput("LungMet", "LungMet", choices = c("Yes", "No"))),
          column(4, selectInput("BoneMet", "BoneMet", choices = c("Yes", "No"))),
          column(4, selectInput("LNMet", "LNMet", choices = c("Yes", "No")))
        )
      )
    ),
    column(
      width = 7,
      wellPanel(
        style = "background-color: #333; border-color: #444; min-height: 600px;", # 保持高度一致
        h4("Survival Probabilities", style = "text-align: center;"),
        tableOutput("resultTable"),
        plotOutput("survivalPlot", height = "400px")
      )
    )
  )
)


server <- function(input, output) {
  dat <- reactive({
    tryCatch({
      df <- read.csv(train_data_f) %>%
        mutate(
          OS_status = as.numeric(OS_status == 1),
          OS_time = as.numeric(OS_time)
        )
      if (nrow(df) <= 0) {
        validate(need(FALSE, "ERROR: Empty dataset!"))
      }
      return(df)
    }, error = function(e) {
      validate(need(FALSE, paste("Data loading error:", e$message)))
    })
  })

  cox_model <- reactive({
    req(dat())
    tryCatch({
      surv_obj <- Surv(dat()$OS_time, dat()$OS_status)
      model <- cph(surv_obj ~ MMF, data = dat(), x = TRUE, y = TRUE, surv = TRUE)
      return(model)
    }, error = function(e) {
      validate(need(FALSE, paste("Model error:", e$message)))
    })
  })

  surv_pred <- reactive({
    req(cox_model())
    patient_input <- data.frame(
      Ensemble_DL = as.numeric(input$Ensemble_DL),
      age = as.numeric(factor(input$age, levels = c("<=55", ">55"))) - 1,
      Cirrhosis = as.numeric(factor(input$Cirrhosis, levels = c("No", "Yes"))) - 1,
      Child_Pugh = as.numeric(factor(input$Child_Pugh, levels = c("A", "B"))) - 1,
      AFP_level = as.numeric(factor(input$AFP_level, levels = c("<=400", ">400"))) - 1,
      BCLC_stage = as.numeric(factor(input$BCLC_stage, levels = c("C", "B"))) - 1,
      ECOG_PS = as.numeric(factor(input$ECOG_PS, levels = c("0", "1"))) - 1,
      max_tumor_diameter = as.numeric(factor(input$max_tumor_diameter, levels = c("<=10", ">10"))) - 1,
      Up_to_Seven = as.numeric(factor(input$Up_to_Seven, levels = c("Yes", "No"))) - 1,
      PVTT = as.numeric(factor(input$PVTT, levels = c("0", "1", "2"))) - 1,
      Line_of_ICIs = as.numeric(factor(input$Line_of_ICIs, levels = c("1", "2"))) - 1,
      LungMet = as.numeric(factor(input$LungMet, levels = c("No", "Yes"))) - 1,
      BoneMet = as.numeric(factor(input$BoneMet, levels = c("No", "Yes"))) - 1,
      LNMet = as.numeric(factor(input$LNMet, levels = c("No", "Yes"))) - 1,
      Treatment = 0
    )

    mmf1 <- predict_risk(rsf_model, patient_input)
    patient_input$Treatment <- 1
    mmf2 <- predict_risk(rsf_model, patient_input)
    print(c(mmf1, mmf2))
    patients <- data.frame(
      MMF = c(mmf1/100, mmf2/100),
      Group = c("ICI+MTT", "ICI+MTT+TACE")
    )
    time_points <- seq(0, 58, by = 1)
    pred_list <- list()
    for(i in 1:nrow(patients)){
      fit <- survest(cox_model(), newdata = patients[i, ], times = time_points, conf.int = 0.95)
      pred_list[[i]] <- data.frame(
        Time = c(0, fit$time),
        Survival = c(1, fit$surv),
        Lower = c(1, fit$lower),
        Upper = c(1, fit$upper),
        Group = patients$Group[i]
      )
    }
    bind_rows(pred_list)
  })

  output$resultTable <- renderTable({
    req(surv_pred())
    df <- surv_pred() %>%
      filter(abs(Time - 24) < 0.5 | abs(Time - 36) < 0.5) %>%
      mutate(
        Time = ifelse(abs(Time - 24) < 0.5, "24-month", "36-month"),
        CI = sprintf("%.4f (%.4f - %.4f)", Survival, Lower, Upper)
      ) %>%
      select(Group, Time, CI) %>%
      pivot_wider(names_from = Group, values_from = CI) %>%
      rename(`Time Point` = Time) %>%
      mutate(Result = case_when(
        `Time Point` == "24-month" ~ "Predicted 2-year survival (95% CI)",
        `Time Point` == "36-month" ~ "Predicted 3-year survival (95% CI)"
      )) %>%
      select(Result, `ICI+MTT`, `ICI+MTT+TACE`) %>%
      rename("ICI + MTT" = "ICI+MTT", "ICI + MTT + TACE" = "ICI+MTT+TACE")
    df
  }, align = "c", bordered = TRUE, width = "100%")

  output$survivalPlot <- renderPlot({
    req(surv_pred())
    color_palette <- c("#D4562E", "#7AB656")

    ggplot(surv_pred(), aes(x = Time, y = Survival, color = Group, fill = Group)) +
      geom_step(linewidth = 1.5) +
      geom_ribbon(aes(ymin = Lower, ymax = Upper), alpha = 0.2, colour = NA) +
      geom_vline(xintercept = c(12, 24, 36, 48), linetype = "dashed", color = "gray70", alpha = 0.6) +
      scale_x_continuous(
        breaks = seq(0, 59, 6),
        expand = expansion(mult = c(0, 0))
      ) +
      scale_y_continuous(
        limits = c(0, 1.01),
        expand = expansion(mult = c(0, 0))
      ) +
      scale_color_manual(values = color_palette, labels = c("ICI + MTT", "ICI + MTT + TACE"), name = "") +
      scale_fill_manual(values = color_palette, labels = c("ICI + MTT", "ICI + MTT + TACE"), name = "") +
      labs(
        title = "Predicted Survival Curves",
        x = "Time (months)",
        y = "Survival Probability"
      ) +
      theme_bw(base_size = 14) +
      theme(
        plot.title = element_text(hjust = 0.5, size = 24, face = "bold", color = "white"),
        axis.title = element_text(face = "bold", color = "black"),
        panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
        panel.grid = element_blank(),
        panel.background = element_rect(fill = "white"),
        plot.background = element_rect(fill = "white"),
        axis.ticks = element_line(color = "black"),
        axis.text = element_text(color = "black", size = 12, face = "bold"),
        legend.text = element_text(color = "black", size = 14, face = "bold"),
        legend.direction = "vertical",
        legend.position = "inside",
        legend.position.inside = c(0.65, 0.85),
        legend.justification = c(0, 0),
        legend.title = element_blank(),
        legend.background = element_blank(),
      ) +
      coord_cartesian(ylim = c(0, 1), xlim = c(0, 60))
  })
}

shinyApp(ui = ui, server = server)

