# Importer la bibliothèque OpenCV
import cv2,time

# Essayer de capturer la caméra
try:
  video=cv2.VideoCapture(0)
except:
  print("Erreur : impossible d'accéder à la caméra")
  exit()

# Initialiser la première image de référence
first_frame= None

# Boucle principale
while True:
  # Lire une image depuis la caméra
  check,frame=video.read()
  
  # Vérifier si l'image est valide
  if check:
    # Convertir l'image en niveaux de gris
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # Appliquer un flou gaussien pour réduire le bruit
    gray=cv2.GaussianBlur(gray,(21,21),0)
    
    # Si la première image n'est pas définie, l'utiliser comme référence
    if first_frame is None:
      first_frame=gray
      continue
    
    # Calculer la différence absolue entre l'image courante et la première image
    delta_frame=cv2.absdiff(first_frame,gray)
    # Appliquer un seuil pour obtenir une image binaire
    threshold_frame=cv2.threshold(delta_frame,50,255,cv2.THRESH_BINARY)[1]
    # Dilater l'image binaire pour combler les trous
    threshold_frame=cv2.dilate(threshold_frame,None,iterations=2)
    
    # Trouver les contours des zones en mouvement
    (cntr,_)=cv2.findContours(threshold_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    # Dessiner des rectangles autour des zones en mouvement
    for contour in cntr:
      # Ignorer les zones trop petites
      if cv2.contourArea(contour)<25000:
        continue
      
      # Calculer le rectangle englobant le contour
      (x,y,w,h)=cv2.boundingRect(contour)
      # Dessiner le rectangle sur l'image originale
      cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    
    # Afficher l'image originale avec les rectangles
    cv2.imshow('Frame',frame)
    
    # Attendre une touche du clavier
    key=cv2.waitKey(1)
    
    # Quitter la boucle si la touche q est pressée
    if key == ord('q'):
      break
  
  else:
    print("Erreur : image invalide")
    break

# Libérer la caméra
video.release()
# Détruire les fenêtres d'affichage
cv2.destroyAllWindows()
