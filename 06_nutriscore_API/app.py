import gradio as gr
import pickle
import pandas as pd
from utils import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


with open('./model.pickle', 'rb') as model_file:
  pipeline = pickle.load(model_file)


def image_score(score):
  if score == 'a':
    return mpimg.imread('./images/Nutriscore_A.png')
  if score == 'b':
    return mpimg.imread('./images/Nutriscore_B.png')
  if score == 'c':
    return mpimg.imread('./images/Nutriscore_C.png')
  if score == 'd':
    return mpimg.imread('./images/Nutriscore_D.png')
  if score == 'e':
    return mpimg.imread('./images/Nutriscore_E.png') 


def greet(energy, saturated_fats, sugars, fibres, proteins, salt):
  """
  This is our main predict function
  """
  data_file = pd.DataFrame(columns={
    'energy-kcal_100g',
    'saturated-fat_100g',
    'sugars_100g',
    'fiber_100g',
    'proteins_100g',
    'salt_100g'
  })

  data_file = data_file.append({
    'energy-kcal_100g':float(energy),
    'saturated-fat_100g':float(saturated_fats),
    'sugars_100g':float(sugars),
    'fiber_100g':float(fibres),
    'proteins_100g':float(proteins),
    'salt_100g':float(salt)
  },ignore_index=True)

  nutrigrade = pipeline.predict(data_file)
  return image_score(nutrigrade[0])

description = (
  "Cette inferface vous donne la possibilité de calculer une estimation "\
  "du nutriscore du produit de votre choix. Pour cela, vous devez vous munir des valeurs\n"\
  "nutritionnelles du produit, qui se trouvent très souvent sur l'arrière du packaging."
)

article = (
    "<h2>Aide à l'utilisation</h2>"+
    '<p><ul><li>Veuillez mettre vos nombres avec des "." et non pas des virgules</li>'+
    '<li>Veuillez remplir toutes les cases. Si un champ est manquant sur votre étiquette, veuillez remplir le champ avec la valeur 0</li>'+
    '<li>Si la valeur "sels" n"est pas disponible, veuillez mettre la valeur sodium * 2.5. Cas échéant mettre la valeur 0</li>'+
    '<li>Veuillez bien choisir les valeurs pour 100g de produit</li></ul></p>'+
    '<br>'+
    "<h2>Informations supplémentaires</h2>"+
    "<p><ul><li>Notre algorithme se base sur les quantités d'energie en kCal, d'acide gras saturés, de sucres, de fibre, de protéines et de sels pour estimer le nutriscore</li>"+
    "<li>Notre analyse repose sur l'hypothèse que ces 6 facteurs sont les composantes principales du Nustriscore</li>"+
    "<li>Nous avons entrainé nos modèles sur un échantillon de 350,000 produits</li>"+
    "<li>Quelques chiffres sur le nutriscore: <a href='https://solidarites-sante.gouv.fr/IMG/pdf/nutriscorebilan3ans.pdf'>Lien</a></li></ul></p>"
)

energy_kcal_100g = gr.inputs.Number(
  label = 'Energy per 100g (in kcal)'
)

saturated_fats = gr.inputs.Number(
  label = 'Saturated fats per 100g (in g)'
)

sugars = gr.inputs.Number(
  label = 'Sugars per 100g (in g)'
)

fibres = gr.inputs.Number(
  label = 'Fibres per 100g (in g)'
)

proteins = gr.inputs.Number(
  label = 'Proteins per 100g (in g)'
)

salt = gr.inputs.Number(
  label = 'Salt per 100g (in g) (Note: Salt = Sodium * 2.5)'
)

image = gr.outputs.Image(
  label = 'Le Nutriscore estimé est:'
)

iface = gr.Interface(
  fn=greet, 
  inputs=[energy_kcal_100g,saturated_fats,sugars,fibres,proteins,salt], 
  outputs=image,
  article = article,
  title = 'Prédiction de Nutriscore',
  description = description,
  allow_flagging='never',
  theme='default'
  )


iface.launch(share=True)
